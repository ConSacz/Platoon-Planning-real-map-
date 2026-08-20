from scipy.io import savemat, loadmat
import os
import tarfile
import pickle
import numpy as np

import importlib

# %% LOAD MAP 
CASE_CONFIG = {
    0: "map Viet Nam",
    1: "map Europe",
    2: "map America"
}


def load_locations(case):

    folder = CASE_CONFIG[case]

    # workspace_functions.py nằm trong Project/utils/
    utils_dir = os.path.dirname(os.path.abspath(__file__))

    # Đi lên Project/
    project_dir = os.path.dirname(utils_dir)

    # Đường dẫn tới locations.py
    locations_path = os.path.join(
        project_dir,
        "map generation",
        folder,
        "locations.py"
    )

    if not os.path.exists(locations_path):
        raise FileNotFoundError(
            f"Không tìm thấy locations.py:\n{locations_path}"
        )

    spec = importlib.util.spec_from_file_location(
        f"locations_case_{case}",
        locations_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.locations

#%%
def save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, runtime):
    os.makedirs(folder_name, exist_ok=True)
    savemat(os.path.join(folder_name, file_name), {
        'ARRIVAL_TIME': ARRIVAL_TIMES,
        'init': init,
        'pop': pop,
        'BestCostIt': BestCostIt,
        'best': best,
        'runtime': runtime
    })
    
#%%
def load_mat(folder_name, file_name):
    # Đảm bảo đường dẫn file tồn tại
    file_path = os.path.join(folder_name, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    
    # Tải dữ liệu từ file .mat với cấu trúc đầy đủ
    data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    # Trích xuất các biến cần thiết
    ARRIVAL_TIMES = data['ARRIVAL_TIME']
    init = data['init']
    pop = data['pop']
    pop_dicts = [matlab_struct_to_dict(item) for item in pop]
    BestCostIt = data['BestCostIt']
    best = matlab_struct_to_dict(data['best'])
    runtime = data['runtime']
    
    # Trả về các biến dưới dạng dictionary
    return {
        'ARRIVAL_TIMES': ARRIVAL_TIMES,
        'init': init,
        'pop': pop_dicts,
        'BestCostIt': BestCostIt,
        'best': best,
        'runtime': runtime
    }

# %% chuyển từ struct kiểu MATLAB qua dict kiểu Python
def matlab_struct_to_dict(struct):
    return {field: getattr(struct, field) for field in struct._fieldnames}

# %% load spyder data
def load_spydata(filename):
    """
    Load Spyder .spydata saved as TAR archive with pax headers.
    """
    with tarfile.open(filename, "r") as tar:
        members = tar.getmembers()

        # tìm file pickle bên trong
        for m in members:
            if m.name.endswith(".pickle") or m.name.endswith(".pkl"):
                f = tar.extractfile(m)
                obj = pickle.load(f)
                if isinstance(obj, dict) and "globals" in obj:
                    return obj["globals"]
                return restore_globals(obj)

        raise ValueError("Không tìm thấy file pickle trong spydata.")
        
        
def restore_spyder_array(obj):
    if not isinstance(obj, dict):
        return obj

    # Spyder save-array format
    if '__save_array' in obj or '__restore_array__' in obj:
        shape = obj.get('__shape__') or obj.get('shape')
        dtype = obj.get('__dtype__') or obj.get('dtype')
        data = obj.get('__data__') or obj.get('data')

        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)

    return obj

def restore_globals(globals_dict):
    restored = {}
    for k, v in globals_dict.items():
        restored[k] = restore_spyder_array(v)
    return restored

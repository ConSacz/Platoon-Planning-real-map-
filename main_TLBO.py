# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
import numpy as np
import time

from utils.fitness_functions import weighted_fitness
from utils.workspace_functions import save_mat, load_locations
from utils.algorithm_functions import init_individual, copy_ind, population_mean, teaching_phase, learner_phase
import pickle

# FITNESS FUNCTION
def fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary):
    return weighted_fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)


region_set = ["map Viet Nam","map Europe","map America"]
map_ID = 1
region = region_set[map_ID]

with open(f"map generation/{region}/route_library.pkl","rb") as f: 
    RouteLibrary = pickle.load(f)
    
print(f"simulation on {region}")
# =========================================================
# %%PARAMETERS

route_options = len(next(iter(RouteLibrary.values())))

N_trans = 0
for (origin, destination), routes in RouteLibrary.items():
    for route_id, route in routes.items():
        n_nodes = len(route["path"])
        if n_nodes > N_trans:
            N_trans = n_nodes

del destination, n_nodes, origin, route, route_id, routes

# N_set = [60, 80, 100]
N_set = [60]
for N in N_set:
    for trial in range(1):
        
        np.random.seed(trial)
        
        # N = 100
        POP_SIZE = 100
        MaxIt = 250
        
        TIME_WINDOW = (0, 48)
        ARRIVAL_TIMES = np.random.randint(0, 6, N)
        max_wait = 4
        
        locations = load_locations(map_ID)

        ORIGINS = [
            name for name, info in locations.items()
            if info["type"] == "start"
        ]
        
        DESTINATIONS = [
            name for name, info in locations.items()
            if info["type"] == "destination"
        ]
        
        origin_idx = np.random.randint(0, len(ORIGINS), N)
        destination_idx = np.random.randint(0, len(DESTINATIONS), N)
        
        init = [
            (ORIGINS[o], DESTINATIONS[d])
            for o, d in zip(origin_idx, destination_idx)
        ]
        
        BestCostIt = np.zeros(MaxIt)
        
        del origin_idx, destination_idx
        
        # %%MAIN

        pop = [
            init_individual(N, route_options, N_trans, max_wait)
            for _ in range(POP_SIZE)
        ]
        
        best = min(
            pop,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        )
        
        best_fit = fitness(best, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        print(f"Case {N}N, Trial {trial}, Iter 0: {best_fit:.4f}")
        # %% TLBO LOOP
        start_time = time.time()
        for it in range(MaxIt):
            # TEACHER
            teacher = min(
                pop,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            )
        
            mean_ind = population_mean(pop, route_options, max_wait)
        
            # TEACHING PHASE
            for i in range(POP_SIZE):
        
                new_ind = teaching_phase(pop[i], teacher, mean_ind, route_options, max_wait)
        
                fit_old = fitness(pop[i], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                fit_new = fitness(new_ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                if fit_new < fit_old:
                    pop[i] = new_ind
        
            # LEARNER PHASE
            for i in range(POP_SIZE):
        
                j = np.random.randint(0, POP_SIZE-1)
        
                while j == i:
                    j = np.random.randint(0, POP_SIZE-1)
        
                fit_i = fitness(pop[i], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                fit_j = fitness(pop[j], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                new_ind = learner_phase(pop[i], pop[j], fit_i, fit_j, route_options, max_wait)
                fit_new = fitness(new_ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                if fit_new < fit_i:
                    pop[i] = new_ind
            
            # GLOBAL BEST
            current_best = min(
                pop,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            )
        
            current_fit = fitness(current_best, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
            if current_fit < best_fit:
                best = copy_ind(current_best)
                best_fit = current_fit
            BestCostIt[it] = best_fit
            print(f"Case {N}N, Trial {trial}, Iter {it}: {best_fit:.4f}")
        total_time = (time.time() - start_time)/60
        
        folder_name = f'data/case_{N}/TLBO'
        file_name = f'TLBO_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        del current_best, current_fit, it, fit_i, fit_j, fit_new, fit_old, i, j, new_ind
        del MaxIt, mean_ind, POP_SIZE, start_time, teacher

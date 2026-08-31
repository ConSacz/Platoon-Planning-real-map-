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
from utils.algorithm_functions import init_individual, copy_ind, population_mean, Levy, clamp_individual
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
        
        q = 0.1
        e = 2.2204e-16
        D = 1.5
        
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
            mean_ind = population_mean(pop, route_options, max_wait)
            # PDO parameters
            # r alternates between -1 and +1
            r = -1 if it% 2 == 1 else 1
            # Digging strength DS
            DS = 1.5 * r * (1 - it/ MaxIt) ** (2 * it/ MaxIt)
            # Predator effect PE
            PE = 1.5 * (1 - it/ MaxIt) ** (2 * it/ MaxIt)

            for i in range(POP_SIZE):
                j = np.random.randint(POP_SIZE)
                ind = copy_ind(pop[i])
                fi = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                
                CBest = {}
                CBest["route"] = best["route"] * D + (ind["route"] * mean_ind["route"]) / (best["route"] * route_options + D)
                CBest["wait"] = best["wait"] * D + (ind["wait"] * mean_ind["wait"]) / (best["wait"] * max_wait + D)
                
                CPD = {}
                CPD["route"] = (best["route"] - pop[j]["route"]) / (best["route"] + D)
                CPD["wait"] = (best["wait"] - pop[j]["wait"]) / (best["wait"] + D)
                
                levy_route = Levy(1,N)
                levy_wait = Levy(N_trans - 1, N)
                
                if it < MaxIt / 4:
                    ind["route"] = (best["route"] - CBest["route"] * q - CPD["route"] * levy_route)
                    ind["wait"] = (best["wait"] - CBest["wait"] * q - CPD["wait"] * levy_wait)
                    
                elif it < MaxIt / 2:
                    ind["route"] = (best["route"] * pop[j]["route"] * DS * levy_route)
                    ind["wait"] = (best["wait"] * pop[j]["wait"] * DS * levy_wait)
                    
                elif it < 3 * MaxIt / 4:
                    rand_route = np.random.random(N)
                    rand_wait = np.random.random((N_trans - 1, N))
                    ind["route"] = (best["route"] - CBest["route"] * e - CPD["route"] * rand_route)
                    ind["wait"] = (best["wait"] - CBest["wait"] * e - CPD["wait"] * rand_wait)
                    
                else:
                    rand_route = np.random.random(N)
                    rand_wait = np.random.random((N_trans - 1, N))
                    ind["route"] = (best["route"] * PE * rand_route)
                    ind["wait"] = (best["wait"] * PE * rand_wait)
            
            ind = clamp_individual(ind, route_options, max_wait)
            ind_cost = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            if ind_cost <= fi:
                pop[i] = copy_ind(ind)
                if ind_cost < best_fit:
                    best = ind
                    best_fit = ind_cost
                    
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
        
        del CBest, CPD, current_best, current_fit, DS, e, f, fi, D, i, ind, ind_cost,it, j, levy_route, levy_wait, PE, q, r
        del MaxIt, mean_ind, POP_SIZE, start_time, rand_route, rand_wait

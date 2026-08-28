# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
import numpy as np
import time
from math import pi

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
        # ALGORITHM PARAMETERS
        # N = 100
        POP_SIZE = 100
        MaxIt = 250
        alpha = 0.1
        delta = 0.1
        
        # PROBLEM PARAMETERS
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
            G2 = 2 * np.random.rand() - 1
            G1 = 2 * (1 - (it / MaxIt))
            to = np.arange(1, N + 1)
            u = 0.0265
            r0 = 10
            r = r0 + u * to
            omega = 0.005
            phi0 = 3 * pi / 2
            phi = -omega * to + phi0
            x = r * np.sin(phi)
            y = r * np.cos(phi)
            QF = it ** ((2 * np.random.rand() - 1) / (1 - MaxIt) ** 2)
            
            # TEACHER
            teacher = min(
                pop,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            )
        
            mean_ind = population_mean(pop, route_options, max_wait)
        
            for i in range(POP_SIZE):
                
                # %% EXPLORATION
                if it < 2/3 * MaxIt:
                    if np.random.rand() < 0.5:
                        ind = copy_ind(pop[i])
                        fi = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        
                        ind['route'] = (best['route'] * (1 - it / MaxIt) + (np.mean(ind['route']) - best['route']) * np.random.rand())
                        ind['wait'] = (best['wait'] * (1 - it / MaxIt) + (np.mean(ind['wait']) - best['wait']) * np.random.rand())
                        ind = clamp_individual(ind, route_options, max_wait)
                        
                        ind_cost = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        if ind_cost <= fi:
                            pop[i] = copy_ind(ind)
                            if ind_cost < best_fit:
                                best = ind
                                best_fit = ind_cost
                    else:
                        j = np.random.randint(POP_SIZE)
                        ind = copy_ind(pop[i])
                        fi = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        
                        levy_route = Levy(1,N)
                        levy_wait = Levy(N_trans - 1, N)
                        
                        ind['route'] = (best['route'] * levy_route + pop[j]['route'] + (y - x) * np.random.rand())
                        ind['wait'] = (best['wait'] * levy_wait + pop[j]['wait'] + (y - x) * np.random.rand())
                        ind = clamp_individual(ind, route_options, max_wait)
                        
                        ind_cost = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        if ind_cost <= fi:
                            pop[i] = copy_ind(ind)
                            if ind_cost < best_fit:
                                best = ind
                                best_fit = ind_cost
                                
                # %% EXPLOITATION
                else:
                    if np.random.rand() < 0.5:
                        ind = copy_ind(pop[i])
                        mean_ind = population_mean(pop, route_options, max_wait)
                        fi = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        
                        ind['route'] = ((best['route'] - np.mean(mean_ind['route'])) * alpha - 
                                        np.random.rand() + (np.random.rand(N)) * delta)
                        ind['wait'] = ((best['wait'] - np.mean(mean_ind['wait'])) * alpha - 
                                        np.random.rand() + (np.random.rand(N_trans - 1, N)) * delta)
                        ind = clamp_individual(ind, route_options, max_wait)
                        
                        ind_cost = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        if ind_cost <= fi:
                            pop[i] = copy_ind(ind)
                            if ind_cost < best_fit:
                                best = ind
                                best_fit = ind_cost
                    else:
                        ind = copy_ind(pop[i])
                        fi = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        
                        levy_route = Levy(1,N)
                        levy_wait = Levy(N_trans - 1, N)
                        
                        ind['route'] = (QF * best['route'] - G2 * ind['route'] * np.random.rand() - 
                                        G1 * levy_route + np.random.rand() * G2)
                        ind['wait'] = (QF * best['wait'] - G2 * ind['wait'] * np.random.rand() - 
                                        G1 * levy_wait + np.random.rand() * G2)
                        ind = clamp_individual(ind, route_options, max_wait)
                        
                        ind_cost = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        if ind_cost <= fi:
                            pop[i] = copy_ind(ind)
                            if ind_cost < best_fit:
                                best = ind
                                best_fit = ind_cost
            # %% GLOBAL BEST
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

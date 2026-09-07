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
from utils.algorithm_functions import init_individual, copy_ind, clamp_individual, generate_and_select_best_neighbor
import pickle

# FITNESS FUNCTION
def fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary):
    return weighted_fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)

# =========================================================
# %% MAP GENERATION
region_set = ["map Viet Nam","map Europe","map America"]
map_ID = 1
region = region_set[map_ID]

with open(f"map generation/{region}/route_library.pkl","rb") as f: 
    RouteLibrary = pickle.load(f)
    
print(f"simulation on {region}")

route_options = len(next(iter(RouteLibrary.values())))

N_trans = 0
for (origin, destination), routes in RouteLibrary.items():
    for route_id, route in routes.items():
        n_nodes = len(route["path"])
        if n_nodes > N_trans:
            N_trans = n_nodes

del destination, n_nodes, origin, route, route_id, routes


# %% MAIN
# N_set = [60, 80, 100]
N_set = [60]
for N in N_set:
    for trial in range(1):
        # %%PARAMETERS
        np.random.seed(trial)
        
        # N = 100
        POP_SIZE = 100
        MaxIt = 250
        beta = 8 # number of neighbors
        
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
        
        # %% INIT

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
            for i in range(POP_SIZE):
                r = np.random.randint(POP_SIZE) # flow(r)
                
                # %% FDA operators
                xi = copy_ind(pop[i])
                xr = copy_ind(pop[r])
                fi = fitness(xi, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                fr = fitness(xr, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                
                W_global_route = 1.0 * ((1 - it / MaxIt)**(2 * np.random.randn())) * (np.random.rand(N) * it / MaxIt) * np.random.rand(N)
                W_global_wait = 1.0 * ((1 - it / MaxIt)**(2 * np.random.randn())) * (np.random.rand(N_trans - 1, N) * it / MaxIt) * np.random.rand(N_trans - 1, N)
                
                Delta_g = copy_ind(pop[i])
                Delta_g['route'] = (np.random.rand(N) * xr['route'] - 
                                    np.random.rand(N) * xi['route']) * np.linalg.norm(best['route'] - xi['route']) * W_global_route
                Delta_g['wait'] = (np.random.rand(N_trans - 1, N) * xr['wait'] - 
                                   np.random.rand(N_trans - 1, N) * xi['wait']) * np.linalg.norm(best['wait'] - xi['wait']) * W_global_wait
                
                # Generate and select best neighbor
                best_neighbor, best_neighbor_cost = generate_and_select_best_neighbor(
                    xi, Delta_g, beta, init, ARRIVAL_TIMES, route_options, N_trans, max_wait, RouteLibrary)
                
                # %% Neighbor is better than current flow(i)
                if best_neighbor_cost <= fi:
                    ind = copy_ind(xi)

                    V_route = np.random.randn(N) * (best_neighbor['route'] - xi['route'])
                    V_wait = np.random.randn(N_trans - 1, N) * (best_neighbor['wait'] - xi['wait'])
                    
                    ind['route'] = xi['route'] + V_route
                    ind['wait'] = xi['wait'] + V_wait
                    ind = clamp_individual(ind, route_options, max_wait)
        
                    ind_cost = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                    if ind_cost <= fi:
                        pop[i] = copy_ind(ind)
                        if ind_cost < best_fit:
                            best = ind
                            best_fit = ind_cost
                
                # %% Neighbor is worse than current flow(i)
                else:    
                    # Random flow(r) is better than flow(i)
                    if fr <= fi:
                        ind = copy_ind(xi)
                        ind['route'] = xi['route'] + np.random.randn(N) * (xr['route'] - xi['route'])
                        ind['wait'] = xi['wait'] + np.random.randn(N_trans - 1, N) * (xr['wait'] - xi['wait'])
                        ind = clamp_individual(ind, route_options, max_wait)
                        
                        ind_cost = fitness(xi, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        if ind_cost <= fi:
                            pop[i] = copy_ind(ind)
                            if ind_cost < best_fit:
                                best = ind
                                best_fit = ind_cost
                    
                    # Random flow(r) is worse than flow(i)
                    else:
                        ind = copy_ind(xi)
                        ind['route'] = xi['route'] + 2 * np.random.randn(N) * (xr['route'] - xi['route'])
                        ind['wait'] = xi['wait'] + 2 * np.random.randn(N_trans - 1, N) * (xr['wait'] - xi['wait'])
                        ind = clamp_individual(ind, route_options, max_wait)
                        
                        ind_cost = fitness(xi, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                        if ind_cost <= fi:
                            pop[i] = copy_ind(ind)
                            if ind_cost < best_fit:
                                best = ind
                                best_fit = ind_cost
                                
            print(f"Case {N}N, Trial {trial}, Iter {it}: {best_fit:.4f}")
        total_time = (time.time() - start_time)/60
        
        folder_name = f'data/case_{N}/FDA'
        file_name = f'FDA_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        del i, ind, ind_cost, it, best_neighbor, best_neighbor_cost, beta, Delta_g, f, fi, fr, r
        del MaxIt, POP_SIZE, start_time, V_route, V_wait, W_global_route, W_global_wait, xi, xr

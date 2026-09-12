# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
import numpy as np
import time
import pickle

from utils.fitness_functions import weighted_fitness
from utils.workspace_functions import save_mat, load_locations
from utils.algorithm_functions import PSO_init_individual, update_particle

# FITNESS FUNCTION
def fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary):
    return weighted_fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)

# %% MAP GENERATION
map_ID = 1
N_set = [1000]
# N_set = [60, 80, 100]
Trial = 50

region_set = ["map Viet Nam","map Europe","map America"]
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
for N in N_set:
    for trial in range(Trial):
        np.random.seed(trial)

        POP_SIZE = 100
        MaxIt = 250
        W  = 0.7
        C1 = 1.5
        C2 = 1.5
        
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
        ARRIVAL_TIMES = np.random.randint(0, round(N/(len(ORIGINS)*len(DESTINATIONS))/2), N)
        
        init = [
            (ORIGINS[o], DESTINATIONS[d])
            for o, d in zip(origin_idx, destination_idx)
        ]
        
        BestCostIt = np.zeros(MaxIt)
        
        del origin_idx, destination_idx
        # ---- init pop ----
        pop = [
            PSO_init_individual(N, route_options, N_trans, max_wait)
            for _ in range(POP_SIZE)
        ]
        
        # pbest
        pbest = []
        
        for ind in pop:
            pbest.append({
                "route": ind["route"].copy(),
                "wait": ind["wait"].copy(),
                "prior": ind["prior"].copy()
            })
        
        # gbest
        gbest = min(
            pbest,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        )
        print(f"Case {N}N, Trial {trial}, Iter 0: {fitness(gbest, init, ARRIVAL_TIMES, N_trans, RouteLibrary):.4f}")
        # %% PSO LOOP
        start_loop = time.time()
        for it in range(1,MaxIt+1):
        
            for idx, ind in enumerate(pop):
                ind = update_particle(ind, pbest[idx], gbest, W, C1, C2, route_options, max_wait)
                
                current_fit = fitness( ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                pbest_fit = fitness( pbest[idx], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                # update pbest
                if current_fit < pbest_fit:
                    pbest[idx] = {
                        "route": ind["route"].copy(),
                        "wait": ind["wait"].copy(),
                        "prior": ind["prior"].copy()
                    }
        
            # update gbest
            candidate = min(
                pbest,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            )
        
            if (fitness(candidate, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                <
                fitness(gbest, init, ARRIVAL_TIMES, N_trans, RouteLibrary)):
                gbest = candidate.copy()
            
            BestCostIt[it] = fitness(gbest, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            
            print(f"Case {N}N, Trial {trial}, Iter {it}: {fitness(gbest, init, ARRIVAL_TIMES, N_trans, RouteLibrary):.3f}")
        total_time = (time.time() - start_loop)/60
        print (f"runtime: {total_time:.4f}min")
        
        folder_name = f'data/{region}/case_{N}/PSO'
        file_name = f'PSO_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, gbest, total_time)
        
        # del C1, C2, candidate, current_fit, idx, it, MaxIt, ind, pbest, pbest_fit
        # del start_loop, POP_SIZE, TIME_WINDOW, W

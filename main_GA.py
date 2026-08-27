# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
import numpy as np
import time
import pickle

from utils.algorithm_functions import crossover, mutate, selection_etilist, init_individual
from utils.fitness_functions import weighted_fitness
from utils.workspace_functions import save_mat, load_locations

# FITNESS FUNCTION
def fitness(ind, init, ARRIVAL_TIMES,  N_trans, RouteLibrary):
    return weighted_fitness(ind, init, ARRIVAL_TIMES,  N_trans, RouteLibrary)


region_set = ["map Viet Nam","map Europe","map America"]
map_ID = 2
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
# %%PARAMETERS
N_set = [60, 80, 100]
for N in N_set:
    for trial in range(50):
        
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
        # ---- init pop ----
        pop = [
            init_individual(N, route_options, N_trans, max_wait)
            for _ in range(POP_SIZE)
        ]
        
        best = min(
            pop,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        )
        
        best_fit = fitness(best, init, ARRIVAL_TIMES,  N_trans, RouteLibrary)
        # %%MAIN (GA LOOP HERE)
        

        
        start_loop = time.time()
        for it in range(MaxIt):
            # ---- crossover + mutation ----
            next_pop = []
            for i in range(0, POP_SIZE):
                k = np.random.randint(0,N-1)
                p1, p2 = pop[i], pop[k]
                c1, c2 = crossover(p1, p2)
                next_pop.append(mutate(c1, route_options, N_trans, max_wait))
                next_pop.append(mutate(c2, route_options, N_trans, max_wait))
        
            pop = next_pop
            pop = selection_etilist(pop, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            
            # ---- evaluation ----
            best_fit = min(pop, key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary))
        
            if best is None or fitness(best_fit, init, ARRIVAL_TIMES,  N_trans, RouteLibrary) < fitness(best, init, ARRIVAL_TIMES,  N_trans, RouteLibrary):
                best = best_fit
            BestCostIt[it] = fitness(best, init, ARRIVAL_TIMES,  N_trans, RouteLibrary)
            
            del best_fit, next_pop, c1, c2, p1, p2, k, i
            print(f"Case {N}N, Trial {trial}, Iter {it}: {fitness(best, init, ARRIVAL_TIMES, N_trans, RouteLibrary):.3f}")
        total_time = (time.time() - start_loop)/60
        
        folder_name = f'data/case_{N}/GA'
        file_name = f'GA_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        del it, MaxIt, N_trans, RouteLibrary, POP_SIZE, start_loop, TIME_WINDOW
# print("Best individual:", best)
# print("Decoded routes:", decode(best, paths_to_B, paths_to_C, DESTINATIONS))
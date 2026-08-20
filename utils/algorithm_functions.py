from utils.fitness_functions import weighted_fitness
import random
import numpy as np

# FITNESS FUNCTION
def fitness(ind, init, ARRIVAL_TIMES,  N_trans, RouteLibrary):
    return weighted_fitness(ind, init, ARRIVAL_TIMES,  N_trans, RouteLibrary)

# %% GENERAL FUNCTIONS
def init_individual(N, route_options, N_trans, max_wait):
    
    route = np.random.randint(0, route_options, size=N)
    wait = np.random.randint(1, max_wait + 1, size=(N_trans - 1, N))
    prior = np.array([np.random.permutation(N) for _ in range(N_trans - 1)])
        
    return {
        "route": route,
        "wait": wait,
        "prior": prior
    }

# COPY
def copy_ind(ind):
    return {
        "route": np.asarray(ind["route"], dtype=float).copy(),
        "wait": np.asarray(ind["wait"], dtype=float).copy(),
        "prior": ind["prior"].copy()
    }

# CLAMP
def clamp_individual(ind, route_options, max_wait):
    ind["route"] = np.round(ind["route"]).astype(int)
    ind["wait"] = np.round(ind["wait"]).astype(int)

    ind["route"] = ind["route"] % route_options
    ind["wait"] = (ind["wait"]-1) % (max_wait) + 1
    
    return ind

# %%GA FUNCTIONS
def selection_etilist(pop, init, ARRIVAL_TIMES,  N_trans, RouteLibrary):
    POP_SIZE = len(pop)
    # sort population theo fitness tăng dần (min tốt hơn)
    sorted_pop = sorted(
        pop,
        key=lambda ind: fitness(ind, init, ARRIVAL_TIMES,  N_trans, RouteLibrary)
    )
    return sorted_pop[:POP_SIZE]

def selection_tournament(pop, init, ARRIVAL_TIMES,  N_trans, RouteLibrary):
    k = 3
    new_pop = []
    for _ in range(len(pop)):
        candidates = random.sample(pop, k)
        best = min(candidates, key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary))
        new_pop.append(best)
    return new_pop

def crossover(p1, p2):
    N = len(p1['route'])
    point = random.randint(0, N-1)

    c1 = {
        "route": np.concatenate((p1["route"][:point], p2["route"][point:])),
        "wait": np.concatenate((p1["wait"][:point], p2["wait"][point:])),
        "prior": p1["prior"],
    }
    
    c2 = {
        "route": np.concatenate((p2["route"][:point], p1["route"][point:])),
        "wait": np.concatenate((p2["wait"][:point], p1["wait"][point:])),
        "prior": p2["prior"],
    }

    return c1, c2

def mutate(ind, route_options, N_trans, max_wait):
    N = len(ind['route'])
    MUT_RATE = 0.2

    if random.random() < MUT_RATE:
        ind = init_individual(N, route_options, N_trans, max_wait)     
    return ind

# %% TLBO FUNCTIONS
# MEAN
def population_mean(pop, route_options, max_wait):
    mean_ind = {
        "route": [],
        "wait": []
    }
    for key in mean_ind.keys():
        arr = np.array([p[key] for p in pop])
        mean_ind[key] = np.mean(arr, axis=0)
        # clamp_individual(mean_ind, route_options, max_wait)
    return mean_ind

# TEACHING PHASE
def teaching_phase(student, teacher, mean_ind, route_options, max_wait):
    TF = np.random.randint(1,2)
    new_ind = copy_ind(student)
    r = np.random.random()
    # route 
    new_ind["route"] = new_ind["route"].astype(float) + (r * (teacher["route"] - TF * mean_ind["route"]))
    # wait
    new_ind["wait"] = new_ind["wait"].astype(float) + (r * (teacher["wait"] - TF * mean_ind["wait"]))
    new_ind = clamp_individual(new_ind, route_options, max_wait)
    return new_ind

# LEARNER PHASE
def learner_phase(ind1, ind2, fit1, fit2, route_options, max_wait):
    new_ind = copy_ind(ind1)
    r = np.random.random()
    if fit1 < fit2:
        new_ind["route"] = new_ind["route"].astype(float) + (r * ( ind1["route"] - ind2["route"]))
        new_ind["wait"] = new_ind["wait"].astype(float) + (r * (ind1["wait"] - ind2["wait"]))
    else:
        new_ind["route"] = new_ind["route"].astype(float) + (r * ( ind2["route"] -ind1["route"]))
        new_ind["wait"] = new_ind["wait"].astype(float) + (r * (ind2["wait"] - ind1["wait"]))
    new_ind =clamp_individual(new_ind, route_options, max_wait)
    return new_ind

# %% PSO FUNCTIONS

# PSO INIT
def PSO_init_individual(N, route_options, N_trans, max_wait):
    
    route = np.random.randint(0, route_options, size=N)
    wait = np.random.randint(1, max_wait + 1, size=(N_trans - 1, N))
    prior = np.array([np.random.permutation(N) for _ in range(N_trans - 1)])
        
    return {
        "route": route,
        "wait": wait,
        "prior": prior,
        "v_route": np.random.uniform(-1,1, size=(N)),
        "v_wait": np.random.uniform(-1,1, size=(N_trans - 1, N))
    }

# PSO UPDATE PARTICLE
def update_particle(ind, pbest, gbest, W, C1, C2, route_options, max_wait):
    N = len(ind['route'])
    new_ind = ind.copy()
    for i in range(N):
        r1 = random.random()
        r2 = random.random()
        
        # route 
        v = (
            W * new_ind["v_route"][i]
            + C1 * r1 * (pbest["route"][i] - new_ind["route"][i])
            + C2 * r2 * (gbest["route"][i] - new_ind["route"][i])
        )
        new_ind["v_route"][i] = v
        new_ind["route"][i] = new_ind["route"][i] + v
        
        # wait
        v = (
            W * new_ind["v_wait"][:, i]
            + C1 * r1 * (pbest["wait"][:, i] - new_ind["wait"][:, i])
            + C2 * r2 * (gbest["wait"][:, i] - new_ind["wait"][:, i])
        )
        new_ind["v_wait"][:, i] = v
        new_ind["wait"][:, i] = new_ind["wait"][:, i] + v
    new_ind = clamp_individual(new_ind, route_options, max_wait)
    return new_ind
    
    
    
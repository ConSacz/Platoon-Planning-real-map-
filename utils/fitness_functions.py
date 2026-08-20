from utils.platoon_functions import decode_routes, build_segment_index, build_platoons, build_priority_rank
import numpy as np

# ======================
# FITNESS
# ======================

FCF = np.array([240.525, 223.207, 214.926, 209.575, 206.678]) # fuel cost factors (g/km)
vel = 60 # (km/h)

gas_price = 1.1144/750 #($/L : g/L)
wait_price = 2  # $/hour

# %% separate fitness
def plat_fitness(vehicles, platoon_info):
    N = len(vehicles)

    fuel_cost = 0.0
    wait_cost = 0.0
    cost_matrix = np.zeros(N)

    for veh in vehicles:

        vid = veh["vehicle_id"]

        # ------------------------------------------
        # Fuel
        # ------------------------------------------

        for node_idx, distance in enumerate(veh["segment_distance"]):

            info = platoon_info[(vid, node_idx)]

            pos = info["position"]

            if pos >= len(FCF):
                pos = 0

            fuel = distance * vel * FCF[pos] * gas_price

            fuel_cost += fuel
            cost_matrix[vid] += fuel

        # ------------------------------------------
        # Waiting
        # ------------------------------------------

        wait = np.sum(veh["wait"]) * wait_price

        wait_cost += wait
        cost_matrix[vid] += wait

    return fuel_cost, wait_cost, cost_matrix

# %% original fitness
def ori_fitness(vehicles, platoon_info):
    N = len(vehicles)

    fuel_cost = 0.0
    wait_cost = 0.0
    cost_matrix = np.zeros(N)

    for veh in vehicles:

        vid = veh["vehicle_id"]

        # ------------------------------------------
        # Fuel
        # ------------------------------------------

        for node_idx, distance in enumerate(veh["segment_distance"]):

            info = platoon_info[(vid, node_idx)]

            pos = info["position"]

            if pos >= len(FCF):
                pos = 0

            fuel = distance * vel * FCF[pos] * gas_price

            fuel_cost += fuel
            cost_matrix[vid] += fuel

        # ------------------------------------------
        # Waiting
        # ------------------------------------------

        wait = np.sum(veh["wait"]) * wait_price

        wait_cost += wait
        cost_matrix[vid] += wait

    return fuel_cost, wait_cost, cost_matrix

# %% separate fitness
def sep_fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary):
    vehicles = decode_routes(ind, init, ARRIVAL_TIMES, RouteLibrary)
    segment_index = build_segment_index(vehicles)
    priority_rank = build_priority_rank(ind, N_trans)
    platoon_info = build_platoons(vehicles, segment_index, priority_rank)
    fuel_cost, wait_cost, cost_matrix = plat_fitness(vehicles, platoon_info)
    
    return fuel_cost, wait_cost, cost_matrix

# %% weighted fitness
def weighted_fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary):
    vehicles = decode_routes(ind, init, ARRIVAL_TIMES, RouteLibrary)
    segment_index = build_segment_index(vehicles)
    priority_rank = build_priority_rank(ind, N_trans)
    platoon_info = build_platoons(vehicles, segment_index, priority_rank)
    fuel_cost, wait_cost, cost_matrix = plat_fitness(vehicles, platoon_info)
    
    mean = np.mean(cost_matrix)
    std = np.std(cost_matrix, ddof=1)      # sample standard deviation
    percent_error = std / mean
    return (1 * fuel_cost + 1 * wait_cost) / (1-percent_error)


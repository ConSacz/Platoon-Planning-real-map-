import numpy as np
from collections import defaultdict
# ==========================================================
# %%Decode route + compute timeline
# ==========================================================

def decode_routes(ind, init, ARRIVAL_TIMES, RouteLibrary):

    N = len(ind["route"])

    vehicles = []

    for i in range(N):

        # --------------------------------------------------
        # Route information
        # --------------------------------------------------

        origin = init[i][0]
        destination = init[i][-1]

        route_id = ind["route"][i]

        route = RouteLibrary[(origin, destination)][route_id]

        path = route["path"]
        segments = route["segments"]
        segment_distance = route["segment_distance"]
        segment_time = route["segment_time"]

        n_seg = len(segments)
        n_node = len(path)

        # --------------------------------------------------
        # Wait vector
        # only use first n_seg-1 values
        # destination has no waiting
        # --------------------------------------------------

        n_seg = len(segments)

        wait = ind["wait"][:n_seg, i]

        # Priority vector
        priority_order = ind["prior"][:n_seg]

        # Timeline
        arrival = np.zeros(n_node)
        departure = np.zeros(n_node)
        arrival[0] = ARRIVAL_TIMES[i]
        if n_node > 1:
            departure[0] = arrival[0] + wait[0]
            departure[0] = round(departure[0])
        else:
            departure[0] = arrival[0]

        # Remaining nodes
        for k in range(1, n_node):
            arrival[k] = departure[k-1] + segment_time[k-1]
            if k < n_seg:
                departure[k] = round(arrival[k] + wait[k])
            else:
                departure[k] = arrival[k]

        # Store
        vehicles.append({
            "vehicle_id": i,
            "origin": origin,
            "destination": destination,
            "route_id": route_id,
            "path": path,
            "segments": segments,
            "segment_distance": segment_distance,
            "segment_time": segment_time,
            "arrival": arrival,
            "departure": departure,
            "wait": wait,
            "priority_order": priority_order

        })

    return vehicles

# ==========================================================
# %%Build segment index
# ==========================================================

def build_segment_index(vehicles):

    segment_index = defaultdict(list)

    for veh in vehicles:

        vehicle_id = veh["vehicle_id"]

        segments = veh["segments"]

        departures = veh["departure"]

        priority_orders = veh["priority_order"]

        distances = veh["segment_distance"]

        travel_times = veh["segment_time"]

        for k, segment in enumerate(segments):

            segment_index[segment].append({

                "vehicle": vehicle_id,

                "node_idx": k,

                "departure": departures[k],

                "priority_order": priority_orders[k],

                "distance": distances[k],

                "travel_time": travel_times[k]

            })

    return segment_index

# ==========================================================
# %%Build platoons
# ==========================================================
def build_priority_rank(ind, N_trans):

    priority_rank = []

    for k in range(N_trans - 1):

        perm = ind["prior"][k]

        rank = np.empty(len(perm), dtype=int)

        rank[perm] = np.arange(len(perm))

        priority_rank.append(rank)

    return priority_rank

def build_platoons(vehicles, segment_index, priority_rank):
    platoon_info = {}
    # Each road segment
    for segment, records in segment_index.items():
        # Group by departure time
        departure_groups = defaultdict(list)
        for rec in records:
            departure_groups[rec["departure"]].append(rec)
        # One platoon for each departure time
        for departure, group in departure_groups.items():

            if len(group) == 1:

                rec = group[0]

                platoon_info[(rec["vehicle"], rec["node_idx"])] = {
                    "size": 1,
                    "position": 0
                }
                continue

            # Priority ranking
            node_idx = group[0]["node_idx"]
            
            group.sort(
                key=lambda x: priority_rank[node_idx][x["vehicle"]]
            )

            # Save platoon info
            size = len(group)

            for pos, rec in enumerate(group):

                platoon_info[(rec["vehicle"], rec["node_idx"])] = {
                    "size": size,
                    "position": pos
                }

    return platoon_info
"""
Route Manager

Manage all feasible routes between predefined
origin-destination pairs.

Author: ...
"""

import pickle
import networkx as nx

# ==========================================================
# GLOBAL
# ==========================================================

RouteLibrary = {}

LogisticsGraph = None

# ==========================================================
# LOAD GRAPH
# ==========================================================

def load_graph(filename="logistics_graph.pkl"):

    global LogisticsGraph

    with open(filename,"rb") as f:

        LogisticsGraph = pickle.load(f)

# ==========================================================
# LOAD ROUTES
# ==========================================================

def load_routes(filename="route_library.pkl"):

    global RouteLibrary

    with open(filename,"rb") as f:

        RouteLibrary = pickle.load(f)

# ==========================================================
# SAVE ROUTES
# ==========================================================

def save_routes(filename="route_library.pkl"):

    with open(filename,"wb") as f:

        pickle.dump(RouteLibrary,f)

# ==========================================================
# REGISTER
# ==========================================================

def register(origin,
             destination,
             route_id,
             path,
             geometry=None):

    if (origin, destination) not in RouteLibrary:
        RouteLibrary[(origin, destination)] = {}

    segments = list(zip(path[:-1], path[1:]))

    segment_distance = []
    segment_time = []

    for u, v in segments:

        if not LogisticsGraph.has_edge(u, v):
            raise ValueError(f"Edge not found: {u} -> {v}")

        edge = LogisticsGraph[u][v]

        segment_distance.append(edge["distance"])
        segment_time.append(edge["travel_time"])

    distance = sum(segment_distance)
    travel_time = sum(segment_time)

    RouteLibrary[(origin, destination)][route_id] = {

        "route_id": route_id,

        "path": path,

        "segments": segments,

        "distance": distance,

        "travel_time": travel_time,

        "segment_distance": segment_distance,

        "segment_time": segment_time,

        "geometry": geometry

    }

# ==========================================================
# EXIST
# ==========================================================

def exist(origin,
          destination,
          route_id):

    return route_id in RouteLibrary[(origin,destination)]

# ==========================================================
# GET ROUTE
# ==========================================================

def get(origin,
        destination,
        route_id):

    return RouteLibrary[(origin,destination)][route_id]

# ==========================================================
# GET PATH
# ==========================================================

def get_path(origin,
             destination,
             route_id):

    return get(origin,destination,route_id)["path"]

# ==========================================================
# NUMBER OF ROUTES
# ==========================================================

def num_routes(origin,
               destination):

    return len(RouteLibrary[(origin,destination)])

# ==========================================================
# COMPUTE DISTANCE
# ==========================================================

def compute_distance(origin,
                     destination,
                     route_id):

    path = get_path(origin,destination,route_id)

    total = 0

    for u,v in zip(path[:-1],path[1:]):

        total += LogisticsGraph[u][v]["distance"]

    return total

# ==========================================================
# COMPUTE TRAVEL TIME
# ==========================================================

def compute_travel_time(origin,
                        destination,
                        route_id):

    path = get_path(origin,destination,route_id)

    total = 0

    for u,v in zip(path[:-1],path[1:]):

        total += LogisticsGraph[u][v]["travel_time"]

    return total

# ==========================================================
# GET GEOMETRY
# ==========================================================

def get_geometry(origin,
                 destination,
                 route_id):

    return get(origin,destination,route_id)["geometry"]

# ==========================================================
# SUMMARY
# ==========================================================

def summary():

    print()

    print("="*60)

    print("Route Library")

    print("="*60)

    for od,routes in RouteLibrary.items():

        print()

        print(od[0],"->",od[1])

        print("Routes :",len(routes))

        for rid,data in routes.items():

            d = compute_distance(od[0],od[1],rid)

            t = compute_travel_time(od[0],od[1],rid)

            print(

                f" Route {rid:02d}",

                f"{d:.1f} km",

                f"{t:.2f} h",

                data["path"]

            )

# ==========================================================
# VALIDATE
# ==========================================================

def validate():

    print()

    print("Checking Route Library ...")

    for od,routes in RouteLibrary.items():

        for rid,data in routes.items():

            path = data["path"]

            ok = True

            for u,v in zip(path[:-1],path[1:]):

                if not LogisticsGraph.has_edge(u,v):

                    ok = False

                    break

            if ok:

                print(f"{od} Route {rid:02d} : OK")

            else:

                print(f"{od} Route {rid:02d} : INVALID")
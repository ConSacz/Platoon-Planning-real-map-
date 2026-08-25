from locations import locations
import route_manager as RM
import networkx as nx
import itertools
import pickle
# ==========================================================
# Load logistics graph
# ==========================================================
with open("logistics_graph.pkl","rb") as f: 
    G = pickle.load(f)
    
RM.load_graph()


# ==========================================================
# Parameters
# ==========================================================

K = 8

# ==========================================================
# Origins - DESTINATIONS
# ==========================================================

ORIGINS = [
    name for name, info in locations.items()
    if info["type"] == "start"
]

DESTINATIONS = [
    name for name, info in locations.items()
    if info["type"] == "destination"
]

# ==========================================================
# K Shortest Paths
# ==========================================================

def get_k_shortest_paths(G, source, target, k=8):
    try:
        paths = nx.shortest_simple_paths(
            G,
            source,
            target,
            weight="distance"
        )

        return list(itertools.islice(paths, k))

    except nx.NetworkXNoPath:

        return []


# ==========================================================
# Build Route Library
# ==========================================================

total_routes = 0

for origin in ORIGINS:

    for destination in DESTINATIONS:

        paths = get_k_shortest_paths(
            G,
            origin,
            destination,
            K
        )

        for route_id, path in enumerate(paths):

            RM.register(
                origin=origin,
                destination=destination,
                route_id=route_id,
                path=path
            )
            total_routes+=1


# ==========================================================
# Save
# ==========================================================

RM.save_routes()

print(f"Generated {total_routes} routes.")
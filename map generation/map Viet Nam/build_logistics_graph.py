import pickle
import numpy as np
import networkx as nx

from locations import locations

distance = np.load("data/distance_matrix.npy")

time = np.load("data/travel_time_matrix.npy")

LG = nx.DiGraph()

# --------------------------------------
# Node Index
# --------------------------------------

names = list(locations.keys())

index = {

    name:i

    for i,name in enumerate(names)

}

# --------------------------------------
# Nodes
# --------------------------------------

for name,info in locations.items():

    lat,lon = info["coord"]

    LG.add_node(

        name,

        type=info["type"],

        latitude=lat,

        longitude=lon

    )

# --------------------------------------
# Allowed Edges
# --------------------------------------

allowed_edges = [

    ("Lao Cai IBC","Ha Noi ICD"),

    ("Lao Cai IBC","Tien Son ICD"),
    
    ("Huu Nghi IBC","Ha Noi ICD"),

    ("Huu Nghi IBC","Tien Son ICD"),

    ("Hai Phong Port","Ha Noi ICD"),

    ("Hai Phong Port","Tien Son ICD"),    
    
    ("Ha Noi ICD","Vinh Logistics Center"),

    ("Ha Noi ICD","Da Nang Port"),

    ("Tien Son ICD","Vinh Logistics Center"),

    ("Tien Son ICD","Da Nang Port"),

    ("Vinh Logistics Center","Da Nang Port"),

    ("Vinh Logistics Center","Quy Nhon Port"),

    ("Da Nang Port","Quy Nhon Port"),
    
    ("Da Nang Port","Song Than ICD"),
    
    ("Da Nang Port","Cat Lai Port"),

    ("Quy Nhon Port","Song Than ICD"),
    
    ("Quy Nhon Port","Cat Lai Port"),

    ("Song Than ICD","Cai Mep Port"),

    ("Song Than ICD","Hub Can Tho"),
    
    ("Song Than ICD","Lao Bao IBC"),

    ("Cat Lai Port","Cai Mep Port"),

    ("Cat Lai Port","Hub Can Tho"),
    
    ("Cat Lai Port","Lao Bao IBC"),
]

# --------------------------------------
# Add Edges
# --------------------------------------

for u,v in allowed_edges:

    i = index[u]

    j = index[v]

    LG.add_edge(

        u,

        v,

        distance=distance[i,j],

        travel_time=time[i,j]

    )

# --------------------------------------
# Save
# --------------------------------------

with open("logistics_graph.pkl","wb") as f:

    pickle.dump(LG,f)

print()

print("Logistics graph created.")

print("Nodes :",LG.number_of_nodes())

print("Edges :",LG.number_of_edges())
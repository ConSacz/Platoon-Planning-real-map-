import route_manager as RM

# ==========================================================
# Load logistics graph
# ==========================================================

RM.load_graph()

# ==========================================================
# Route templates
# ==========================================================

ROUTE_TEMPLATES = {

    0: {
        "hubs": [
            "Da Nang Port"
        ]
    },

    1: {
        "hubs": [
            "Vinh Logistics Center",
            "Da Nang Port"
        ]
    },

    2: {
        "hubs": [
            "Da Nang Port",
            "Quy Nhon Port"
        ]
    },

    3: {
        "hubs": [
            "Vinh Logistics Center",
            "Quy Nhon Port"
        ]
    },

    4: {
        "hubs": [
            "Vinh Logistics Center",
            "Da Nang Port",
            "Quy Nhon Port"
        ]
    }

}

# ==========================================================
# Origins
# ==========================================================

ORIGINS = [
    "Huu Nghi IBC",
    "Hai Phong Port",
    "Lao Cai IBC"
]

ORIGINS_HUB = [
    "Ha Noi ICD",
    "Tien Son ICD"
] 

# ==========================================================
# Destinations
# ==========================================================

DESTINATIONS = [
    "Hub Can Tho",
    "Lao Bao IBC",
    "Cai Mep Port"
]

DESTINATIONS_HUB = [
    "Song Than ICD",
    "Cat Lai Port"
] 

# ==========================================================
# Build Route Library
# ==========================================================

for origin in ORIGINS:
    for destination in DESTINATIONS:
        
        for ido, origin_hub in enumerate(ORIGINS_HUB):
            for idd, destination_hub in enumerate(DESTINATIONS_HUB):
                
                for route_template_id, route in ROUTE_TEMPLATES.items():
        
                    path = [origin] + [origin_hub] + route["hubs"] + [destination_hub] + [destination]
                    
                    route_id = (ido * len(DESTINATIONS_HUB) + idd) * len(ROUTE_TEMPLATES) + route_template_id
                    
                    RM.register(
                        origin=origin,
                        destination=destination,
                        route_id=route_id, 
                        path=path
                    )

RM.save_routes()

print(f"Generated {len(ORIGINS)*len(DESTINATIONS)*len(ROUTE_TEMPLATES)} routes.")
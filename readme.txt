Path-planning and Scheduling optimization using GA, PSO, TLBO, AO, PDO, FDA

-Map: Viet Nam, Europe, America
+ map's nodes, including start, hub and destination nodes are defined in 'locations.py'
+ run 'build_distance_matrix.py' then 'build_logistics_graph' to build transportation graph 'logistic_graph.pkl'
+ run 'route_library.py' to build the route library 'route_library.pkl', including available routes and theirs length and travel time


-Problem: 
+ N trucks (15 tons, 60km/h average velocity) arrives at start nodes in random time windows, assigned to travel to destination nodes.
+ Trucks can decide to depart right after arrive, or wait for other to form platoon, platoon order depends on priority index
+ When arrive transit hubs trucks can wait for other to form platoon, platoon order depends on priority index

-Optimization variable: (truck routes, wait time)

-Fitness functions: 
+ total fuel cost(dollar): depends on distance travel, forming platoon and order in platoon
+ total wait time(dollar): wait time at start or hub

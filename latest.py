# %%
# Import files distance_matrix_1.csv and distance_matrix_2.csv
# Voor foto: ![title](assumptions.png) in markdown cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import * 
import math 
import time
import seaborn as sns


small_instance = False               # kleine dataset
dataset = 1                         # kies 1 of 2

# Import the data with delimeter as ; in stad of comma

if dataset == 1:
    distances = pd.read_excel('data.xlsx', sheet_name='Distance 1', index_col=0)
    demand = pd.read_excel('data.xlsx', sheet_name='Demand 1')

if dataset == 2:
    distances = pd.read_excel('data.xlsx', sheet_name='Distance 2', index_col=0)
    demand = pd.read_excel('data.xlsx', sheet_name='Demand 2')

elif small_instance:
    distances = pd.read_excel('data.xlsx', sheet_name='Distance_small', index_col=0)
    demand = pd.read_excel('data.xlsx', sheet_name='Demand_small')

nodes = list(distances.columns) # first and last nodes are both the depot (0 and 14, for start and end point of the routes
for i in nodes:
    distances[i][i] = 99999   # penalise distance from node to itself so the model does not use these edges
distances[0][nodes[-1]] = 99999      # penalise distance from depot to itself
distances[nodes[-1]][0] = 99999     # penalise distance from depot to itself

# Single Depot - Remove last row and column of distance data:
distances = distances.iloc[:-1]
distances = distances.iloc[:, :-1]
demand = demand.iloc[:-1]


# After loading the data
distances.columns = distances.index
distances


# %% [markdown]
# # Get Vehicle data

# %%
vehicles = pd.read_csv('vehicles_10.csv')
vehicles 


# %%
demand

# %% [markdown]
# # Sets and Parameters

# %%
# differ fleet size and capacity with sensitivity analysis 
n_customer_nodes = distances.shape[0] - 1
p_f = 0.75 # Fuel price


# Create Node and Edge classes 
class Node():

    def __init__(self, nr, pickup_demand, delivery_demand):
        self.nr = nr
        self.pickup_demand = pickup_demand
        self.delivery_demand = delivery_demand

class Edge():

    def __init__(self, origin, destination, length):
        self.origin = origin
        self.destination = destination       
        self.length = length        # (or cost)
        
class Vehicle():

    def __init__(self, id, capacity, range, fuel_consumption, cost):
        self.id = id
        self.Q = capacity              # Capacity
        self.TL = range            # Maximum route length 
        self.f = fuel_consumption
        self.c = cost

# Generate Node and Edge objects from provided benchmark data
        
# Customer nodes N 
N = []
for index, row in demand.iterrows():
    if index in range(1, n_customer_nodes+1):
        node = Node(index, row['Pick-up'], row['Delivery'])
        N.append(node)

# T = [N ∪ {0}], all customer nodes in graph plus depot as start (0), 
T = []
for index, row in demand.iterrows():
    node = Node(index, row['Pick-up'], row['Delivery'])
    T.append(node)

V = []
for index, row in vehicles.iterrows(): 
    vehicle = Vehicle(row['ID'], row['Capacity'], row['Range'], row['Fuel Consumption'], row['Cost'])
    V.append(vehicle)

edges = []
for i in T:
    for j in T:
        origin = i.nr 
        destination = j.nr
        length = distances[i.nr][j.nr]
        edges.append(Edge(origin, destination, length))
 

print(f'\nSet N: \n')
for i in N:
    print(f'Node nr: {i.nr}, Pickup Demand: {i.pickup_demand}, Delivery Demand: {i.delivery_demand}')

print(f'\nSet T: \n')
for i in T:
    print(f'Node nr: {i.nr}, Pickup Demand: {i.pickup_demand}, Delivery Demand: {i.delivery_demand}')

print(f'\nSet V: \n')
for v in V:
    print(f'Vehicle {v.id}: Capacity {v.Q}, Max route length {v.TL}, Fuel Consumption {v.f}, Cost {v.c}')

# %% [markdown]
# # Variables

# %%
%%time
m = Model('VRPSPD_var')
##################### Decision Variables ########################
X = {}          # X_ijv = 1 if vehicle v travels from i to j, 0 otherwise
D = {}          # D_iv = The load remaining to be delivered by vehicle v when departing from node i - non-negative integer 
P = {}          # P_iv = The cumulative load picked by vehicle v when departing from node i, - non-negative integer 
y = {}          # y^v = 1 if vehicle v is used, 0 otherwise. 

# X
for i in T:
    for j in T:
        for v in V:
            L_ij = distances[i.nr][j.nr] 
            X[i.nr, j.nr, v.id] = m.addVar(obj = L_ij * v.f * p_f, lb=0,
                                vtype = GRB.BINARY, name = f'X_{i.nr},{j.nr},{v.id}')

# y            
for v in V:
    y[v.id] = m.addVar(obj = v.c, lb=0,
            vtype = GRB.BINARY, name = f'y^{v.id}')

# D & P
for i in T:
    for v in V:
        # D
        D[i.nr, v.id] = m.addVar(obj = 0, lb=0,
                        vtype=GRB.INTEGER, name = f'D_{i.nr},{v.id}')
        
        # P
        P[i.nr, v.id] = m.addVar(obj = 0, lb=0,
                        vtype=GRB.INTEGER, name = f'P_{i.nr},{v.id}')

m.update()
m.setObjective(m.getObjective(), GRB.MINIMIZE)
# m.ModelSense = GRB.MINIMIZE
# m.setObjective( quicksum(quicksum(quicksum( distances[i][j] * v.f * X[i.nr, j.nr, v.id] for i in T) for j in T) for v in V) 
#                 + quicksum(v.c * y[v.id] for v in V) )

# %% [markdown]
# # Constraints

# %%
%%time
#################################################### Constraints ###########################################
# Big M:
M = 1000000


#################################################### C1 ####################################################
# Stipulates that each customer node must be visited by exactly once.
for j in N:  
    m.addConstr(quicksum(quicksum( X[i.nr, j.nr, v.id] for i in T) for v in V), 
                GRB.EQUAL, 1, name=f'C1_{j.nr}')

#################################################### C1* ####################################################
# Variant constraint, ensures a vehicle does not visit any node if it is not used. 
for v in V:
     m.addConstr(quicksum(quicksum( X[i.nr, j.nr, v.id] for i in T) for j in T), GRB.LESS_EQUAL,
                                    M * y[v.id], name = f'C1*_{v.id}')


#################################################### C2 ####################################################
#  Ensures that if a vehicle is used, it leaves and enters the depot once. 
for v in V:
    m.addConstr( quicksum(X[0, j.nr, v.id] for j in N), GRB.EQUAL, y[v.id], name=f'C2_{v.id}')
    m.addConstr( quicksum(X[j.nr, 0, v.id] for j in N), GRB.EQUAL, y[v.id], name=f'C2*_{v.id}')



#################################################### C3 ####################################################
# Ensures that a vehicle arrives and departs from each node it serves.
for j in T: 
    for v in V:
        # C3
        m.addConstr(quicksum( X[i.nr, j.nr, v.id] for i in T)  
                    - quicksum( X[j.nr, i.nr, v.id] for i in T), GRB.EQUAL, 0, name=f'C3_{j.nr}_{v.id}') 



#################################################### C4 ###################################################
# C4
for i in T: 
    for v in V:
        # Ensures that the load on vehicle v, when departing from node i, is always lower than the vehicle capacity.
        m.addConstr( D[i.nr, v.id] + P[i.nr, v.id], GRB.LESS_EQUAL, v.Q * y[v.id], name=f'C4_{i.nr}{v.id}')



#################################################### C5 & C6 ###############################################  
# Ensures that the total delivery load for a route is placed on the vehicle v, embarking on each trip, at the starting node itself. 
for v in V: 
    # C5
    m.addConstr(D[0, v.id] - (quicksum(quicksum((X[i.nr, j.nr, v.id] * i.delivery_demand) for i in N) for j in T)), 
                GRB.EQUAL,  0, name=f'C5_{v.id}') 

for v in V:          
    # C6 
    m.addConstr(P[0, v.id], GRB.EQUAL, 0, name='C6')
 


#################################################### C7 & C8 ################################################
# Transit load constraints i.e., if arc (i, j) is visited by the vehicle v, 
# then the quantity to be delivered by the vehicle has to decrease by d_j while 
# the quantity picked-up has to increase by p_j
for v in V:
    for i in T: 
        for j in N: 
            # C7
            m.addConstr( (D[i.nr, v.id] - j.delivery_demand - D[j.nr, v.id]) * X[i.nr, j.nr, v.id], GRB.EQUAL, 0, name=f'C7_{i.nr}_{j.nr}_{v.id}')
for v in V:
    for i in T: 
        for j in N: 
            # C8
            m.addConstr( (P[i.nr, v.id] + j.pickup_demand - P[j.nr, v.id]) * X[i.nr, j.nr, v.id], GRB.EQUAL, 0, name=f'C8_{i.nr}_{j.nr}_{v.id}')


####################################################  C9  #####################################################
# Ensures the maximum route length for any vehicle v. 
for v in V:
    m.addConstr(quicksum(quicksum( distances[i.nr][j.nr] * X[i.nr, j.nr, v.id] for j in T) for i in T), GRB.LESS_EQUAL, v.TL * y[v.id], name=f'C9_{v.id}')



#################################################### C10 & C11 ###############################################
for v in V:
    for i in T:
        # C10
        m.addConstr(D[i.nr, v.id], GRB.GREATER_EQUAL, 0, name='C10')

        # C11
        m.addConstr(P[i.nr, v.id], GRB.GREATER_EQUAL, 0, name='C11')


# %% [markdown]
# # Optimization

# %%
m.write('MODEL_var.lp')
# Set time constraint for optimization (5minutes)
m.setParam('TimeLimit', 1200)
# Set gap constraint for optimisation
# m.setParam('MIPgap', 0.05)

m.optimize()
m.write("solution_var.sol")
m.write("testout_var.sol")

status = m.status
if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)

# %% [markdown]
# # Verification

# %%
# Test function for checking feasibility:
def get_routes(V, T):
    route_lengths = {}
    routes = {}
    for v in V:
        if y[v.id].X > 0.1:
            # print(f'\nVehicle {v.id}:')
            routes[v.id] = []
            route_length = 0
            for i in T:
                for j in T:
                    if X[i.nr, j.nr, v.id].X > 0.01:
                        # print(f'  {i.nr} --> {j.nr}') 
                        routes[v.id].append((i.nr, j.nr))
                        route_length += distances[i.nr][j.nr]
            route_lengths[v.id] = route_length

    return routes, route_lengths

def sort_routes(route_dict):
    """
    Sorts every route in the right order. Routes start and end at 0.
    
    :param route_dict: Dictionary containing routes with route IDs as keys
                       and lists of tuples (start, end) as values.
    :return: Dictionary with sorted routes.
    """
    sorted_routes = {}
    for vehicle, route in route_dict.items():
        # Start with an empty route
        sorted_route = []
        # Current location starts at 0 (depot)
        current_location = 0
        
        while route:
            # Find the next step in the route where the first element is the current location
            for i, (start, end) in enumerate(route):
                if start == current_location:
                    # Add this step to the sorted route
                    sorted_route.append((start, end))
                    # Remove this step from the route
                    route.pop(i)
                    # Update the current location to the end of this step
                    current_location = end
                    break
                    
        sorted_routes[vehicle] = sorted_route

    # # print routes
    # for vehicle, route in sorted_routes.items():
    #     # print(f'\nVehicle {vehicle}:')
    #     for (i, j) in route:
    #         if X[i, j, vehicle].X > 0.01:
    #             # print(f'  {i} --> {j}') 
    return sorted_routes

def print_routes(routes):
    for vehicle, route in routes.items():
        print(f'\nVehicle {vehicle}:')
        for (i, j) in route:
            if X[i, j, vehicle].X > 0.01:
                print(f'  {i} --> {j}') 

def double_visit(nodes, node):
    count = 0
    for number in nodes:
        if number == node:
            count += 1
    return True if count > 1 else False

def double_depart(nodes, node):
    count = 0
    for number in nodes:
        if number == node:
            count += 1
    return True if count > 1 else False

def customer_node_double_visit(routes, node):
    count = 0
    for vehicle, route in routes.items():
        for edge in route:
            visited = edge[1]
            if node == visited:
                count += 1
    return True if count > 1 else False

def delivery_and_pickup_demand(vehicle_id, route):
    '''returns the required total amounts to be deliverd and picked up in each route'''
    required_D = 0
    required_P = 0
    for edge in route:
        i = edge[0]
        j = edge[1]
        # print(f'From {i} to {j}')
        # print(f'Delivered = {demand["Delivery"].iloc[i]}')
        # print(f'Picked up = {demand["Pick-up"].iloc[i]}')
        required_D += demand["Delivery"].iloc[i]
        required_P += demand["Pick-up"].iloc[i]
    #     print(f' D_{i}, P_{i} = {required_D}, {required_P}')
    # print(f' D and P: {required_D}, {required_P}')    
    return required_D, required_P

def max_vehicle_load(vehicle_id, route):
    '''returns the maximum load a vehicle has carried in a certain route'''
    max_load = 0
    for edge in route:
        i = edge[0]
        j = edge[1]
        load = D[i, vehicle_id].X + P[i, vehicle_id].X
        j_pickup_demand = demand["Pick-up"].iloc[j]
        # print(f' \nEdge {edge}')
        # print(f' D_{i},{vehicle_id} = {D[i, vehicle_id].X}')
        # print(f' P_{i},{vehicle_id} = {P[i, vehicle_id].X}')
        # print(f'\nC8:')
        # print(f'(P_{i},{vehicle_id} + {j_pickup_demand} - P_{j},{vehicle_id}) * X_{i},{j},{vehicle_id} = 0')
        # print(f'({P[i, vehicle_id].X} + {j_pickup_demand} - {P[j, vehicle_id].X}) * {X[i, j, vehicle_id].X} = 0')
        if load > max_load:
            max_load = load

    return max_load 

def check_subtours_and_double_visits(routes):
    '''Check routes for subtours and double visits, return True or False for feasibility'''
    for vehicle, route in routes.items():
        nodes_to = [edge[1] for edge in route]
        nodes_from = [edge[0] for edge in route]
        # print(f'nodes_to {nodes_to} v={vehicle}')
        # print(f'nodes_from: {nodes_from} v={vehicle}')

        # Check if nodes are departed from more than once by same vehicle
        for node in nodes_from:
            if double_depart(nodes_from, node): 
                print(f'Route Infeasible:\n Node {node} departed from more than once by vehicle {vehicle}.')
                return False

        # Check if nodes are visited more than once by same vehicle 
        for node in nodes_to:
            if double_visit(nodes_to, node):     
                print(f'Route Infeasible:\n Node {node} visited more than once by Vehicle {vehicle}.')
                return False

        # Check if any customer nodes are visited more than once
        for customer_node in N:
            if customer_node_double_visit(routes, customer_node.nr):
                print(f'Route Infeasible:\n Customer node {customer_node.nr} visited more than once.')
                return False
    return True


# %% [markdown]
# # Results

# %%
# Get results:
print(f'Objective Function value: {f_objective}')
routes, route_lengths = get_routes(V, T)

# Check for subtours and double visits:
if check_subtours_and_double_visits(routes) == True:
    # sort routes if the routes are contain no subtours or double visits of customer nodes
    routes = sort_routes(routes) 

# print routes:    
print_routes(routes)
print(f'Route lengths: {route_lengths}')


# Check if the maximum loads and route lengths do not exceed capacity
for vehicle, route in routes.items(): 
    v = next((v for v in V if v.id == vehicle), None)
    
    max_load = max_vehicle_load(vehicle, route)
    distance_travelled = route_lengths[vehicle]
    D_required, P_required = delivery_and_pickup_demand(vehicle, route)
    last_customer_node = route[-1][0]
    D_delivered, P_pickedup = D[0, vehicle].X, P[last_customer_node, vehicle].X
    


    # Test delivery and pickup quantities
    if abs(D_required - D_delivered) > 0.01:
        print(f'INFEASIBLE:\nDelivered quantities ({D_delivered}) by vehicle {vehicle} do not match demand ({D_required})')
        break
    
    if abs(P_required - P_pickedup) > 0.01:
        print(f'INFEASIBLE:\nPicked up quantities ({P_pickedup}) by vehicle {vehicle} do not match demand ({P_required})')
        break

    # Test max load
    if max_load > v.Q:
        print(f'INFEASIBLE:\nVehicle {v.id} exceeded capacity')
        break

    # Test max distance
    if distance_travelled > v.TL:
        print(f'INFEASIBLE:\n\n Vehicle {v.id} exceeded Max Route Length')
        break

    print('\n')
    print(f'Vehicle {vehicle}, Capacity {v.Q}, Max route length {v.TL}:')
    print(f'    Max load carried by Vehicle {vehicle} is {max_load}')
    print(f'    Distance travelled by Vehicle {vehicle} is {route_lengths[vehicle]}')
    print(f'    Loads deliverd and picked-up: {D_delivered}, {P_pickedup}')
    print(f'    Demand for delivery and pickup: {D_required}, {P_required}')

    

# %%
## Sensitivity Analysis

# Sensitivity analysis for fuel price

# def sensitivity_analysis_fuel_price(p_f):
#     m.setObjective(m.getObjective(), GRB.MINIMIZE)
#     m.setObjective( quicksum(quicksum(quicksum( distances[i.nr][j.nr] * v.f * X[i.nr, j.nr, v.id] for i in T) for j in T) for v in V) 
#                     + quicksum(v.c * y[v.id] for v in V) )
#     m.optimize()
#     f_objective = m.objVal
#     print(f'\nObjective Function value for fuel price {p_f}: {f_objective}')
#     routes, route_lengths = get_routes(V, T)
#     if check_subtours_and_double_visits(routes) == True:
#         routes = sort_routes(routes) 
#     print_routes(routes)
#     print(f'Route lengths: {route_lengths}')
#     return f_objective

# # Sensitivity analysis for fuel price
# fuel_prices = [0.5, 0.75, 1.0, 1.25, 1.5]
# results = {}
# for p_f in fuel_prices:
#     results[p_f] = sensitivity_analysis_fuel_price(p_f)

# # Plot results
# plt.plot(list(results.keys()), list(results.values()))
# plt.xlabel('Fuel Price')
# plt.ylabel('Objective Function Value')
# plt.title('Sensitivity Analysis for Fuel Price')
# plt.show()



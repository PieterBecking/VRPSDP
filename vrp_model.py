import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
import seaborn as sns

# Class Definitions
class Node:
    def __init__(self, nr, pickup_demand, delivery_demand):
        self.nr = nr
        self.pickup_demand = pickup_demand
        self.delivery_demand = delivery_demand

class Edge:
    def __init__(self, origin, destination, length):
        self.origin = origin
        self.destination = destination
        self.length = length

class Vehicle:
    def __init__(self, id, capacity, range, fuel_consumption, cost):
        self.id = id
        self.capacity = capacity
        self.range = range
        self.fuel_consumption = fuel_consumption
        self.cost = cost

# Function to load data
def load_data(dataset, small_instance=False):
    if dataset == 1:
        distances = pd.read_excel('data.xlsx', sheet_name='Distance 1', index_col=0)
        demand = pd.read_excel('data.xlsx', sheet_name='Demand 1')
    elif dataset == 2:
        distances = pd.read_excel('data.xlsx', sheet_name='Distance 2', index_col=0)
        demand = pd.read_excel('data.xlsx', sheet_name='Demand 2')
    elif small_instance:
        distances = pd.read_excel('data.xlsx', sheet_name='Distance_small', index_col=0)
        demand = pd.read_excel('data.xlsx', sheet_name='Demand_small')

    nodes = list(distances.columns)
    for i in nodes:
        distances[i][i] = 99999
    distances[0][nodes[-1]] = distances[nodes[-1]][0] = 99999
    distances = distances.iloc[:-1, :-1]
    demand = demand.iloc[:-1]
    return distances, demand

# Function to generate vehicle data
def generate_vehicles_data(num_vehicles):
    np.random.seed(42)
    base_factor = np.random.uniform(1, 3, num_vehicles)
    adjustment = np.random.uniform(-0.3, 0.3)
    capacity = np.clip((base_factor + adjustment) * 100, 100, 200)
    range_km = np.clip(base_factor * 200, 200, 450)
    fuel_consumption = np.clip(1 + (base_factor + adjustment - 1) * 0.75, 1, 1.75)
    cost = np.clip((base_factor + adjustment) * 300, 300, 600)
    vehicles_df = pd.DataFrame({
        'ID': range(1, num_vehicles + 1),
        'Capacity': capacity,
        'Range': range_km,
        'Fuel Consumption': fuel_consumption,
        'Cost': cost
    }).round({'Capacity': -1, 'Range': -1, 'Fuel Consumption': 2, 'Cost': -1})
    return vehicles_df

# Setup model
def setup_model(distances, demand, vehicles):
    m = Model('VRPSPD_var')
    T = [Node(idx, row['Pick-up'], row['Delivery']) for idx, row in demand.iterrows()]
    V = [Vehicle(int(row['ID']), row['Capacity'], row['Range'], row['Fuel Consumption'], row['Cost']) for index, row in vehicles.iterrows()]
    X = {(i.nr, j.nr, v.id): m.addVar(vtype=GRB.BINARY, name=f'X_{i.nr},{j.nr},{v.id}')
         for v in V for i in T for j in T}
    D = {(i.nr, v.id): m.addVar(vtype=GRB.INTEGER, name=f'D_{i.nr},{v.id}') for v in V for i in T}
    P = {(i.nr, v.id): m.addVar(vtype=GRB.INTEGER, name=f'P_{i.nr},{v.id}') for v in V for i in T}
    m.update()
    # Define constraints and objective here as needed
    return m, X, D, P, V, T

# Main function to run the model
def run_vrp_model(dataset=1, small_instance=False, num_vehicles=10):
    distances, demand = load_data(dataset, small_instance)
    vehicles = generate_vehicles_data(num_vehicles)
    model, X, D, P, V, T = setup_model(distances, demand, vehicles)
    model.optimize()
    return model

# Entry point of the script
if __name__ == "__main__":
    model = run_vrp_model()
    if model.Status == GRB.Status.OPTIMAL:
        print('Optimal solution found.')
        # Additional processing and output here

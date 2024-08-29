import pandas as pd
from vrp_model import run_vrp_model
from gurobipy import GRB

def perform_sensitivity_analysis():
    vehicle_counts = [5, 10, 15, 20]  # Example vehicle counts to analyze
    capacities = [100, 150, 200]      # Different capacities to analyze
    results = []

    for count in vehicle_counts:
        for capacity in capacities:
            # Adjust the model to use a specific number of vehicles and capacities
            model = run_vrp_model(num_vehicles=count)
            model.setParam('TimeLimit', 300)  # Set a time limit for the model to solve (optional)
            
            # Set the capacity for all vehicles (this might require adjusting the vrp_model.py to allow this kind of parametrization)
            for v in model.getVars():
                if 'Capacity' in v.VarName:
                    v.setAttr(GRB.Attr.UB, capacity)
            
            model.optimize()

            # Check if a valid solution was found
            if model.Status == GRB.Status.OPTIMAL:
                total_cost = model.ObjVal
            else:
                total_cost = float('inf')  # Use infinity to denote no valid solution found within the time limit

            results.append({
                'Number of Vehicles': count,
                'Capacity': capacity,
                'Total Cost': total_cost
            })

    # Create a DataFrame to hold the results
    result_df = pd.DataFrame(results)
    return result_df

def visualize_results(result_df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(x='Number of Vehicles', y='Total Cost', hue='Capacity', data=result_df)
    plt.title('Impact of Fleet Size and Capacity on Total Cost')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Total Cost')
    plt.show()

if __name__ == "__main__":
    result_df = perform_sensitivity_analysis()
    visualize_results(result_df)

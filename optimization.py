import pulp
import pandas as pd
import numpy as np
from scipy.stats import norm

def solve_deterministic_knapsack(catalog_df: pd.DataFrame, virtual_capacity: float):
    """Helper for Approach 1: Solves standard knapsack using a specific capacity."""
    prob = pulp.LpProblem(f"Ahmads_Gift_{virtual_capacity}", pulp.LpMaximize)
    item_vars = pulp.LpVariable.dicts("Item", catalog_df['name'], cat='Binary')

    prob += pulp.lpSum([row['price'] * item_vars[row['name']] for _, row in catalog_df.iterrows()])
    prob += pulp.lpSum([row['estimated_volume'] * item_vars[row['name']] for _, row in catalog_df.iterrows()]) <= virtual_capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    selected_items = [name for name, var in item_vars.items() if var.value() == 1.0]
    if not selected_items:
        return [], 0
    total_price = sum(catalog_df[catalog_df['name'] == item]['price'].values[0] for item in selected_items)
    
    return selected_items, total_price

def calculate_success_probability(selected_items: list, catalog_df: pd.DataFrame, cov_matrix: pd.DataFrame, max_volume: float = 40.0):
    """Shared Helper: Calculates the exact % chance that a specific combination fits."""
    mu = catalog_df[catalog_df['name'].isin(selected_items)]['estimated_volume'].sum()
    
    if not selected_items:
        return 0, 1.0

    x = np.array([1 if name in selected_items else 0 for name in cov_matrix.index])
    variance = x.T @ cov_matrix.values @ x
    std_dev = np.sqrt(variance)
    
    probability = norm.cdf(max_volume, loc=mu, scale=std_dev)
    return mu, probability

def calculate_probability_search_bounds(min_prob: float, max_prob: float, max_volume: float = 40.0,
                                       est_sigma: float = 1.5, min_cap: float = 30.0,
                                       max_cap: float = 50.0):
    """Estimate the search window of expected volumes that could satisfy a probability range."""
    z_min = norm.ppf(min_prob)
    z_max = norm.ppf(max_prob)

    search_v_max = max_volume - (z_min * est_sigma)
    search_v_min = max_volume - (z_max * est_sigma)

    search_v_max = min(search_v_max, max_cap)
    search_v_min = max(search_v_min, min_cap)

    return search_v_min, search_v_max

def calculate_pareto_frontier(solutions_df: pd.DataFrame):
    """Given a DataFrame of solutions, filter down to the Pareto Frontier."""
    pareto_front = []
    max_prob_seen = -1.0
    
    for index, row in solutions_df.iterrows():
        if row['Probability'] > max_prob_seen:
            pareto_front.append(row)
            max_prob_seen = row['Probability']
            
    pareto_df = pd.DataFrame(pareto_front)
    
    # Sort the final Pareto front by Probability descending
    pareto_df = pareto_df.sort_values(by=['Probability', 'Total_Value'], ascending=[False, False])
    
    return pareto_df

def generate_k_best_profiles(catalog_df: pd.DataFrame, cov_matrix: pd.DataFrame, max_volume: float = 40.0, min_prob: float = 0.01, max_prob: float = 0.9999, max_k: int = 500):
    """Approach 2: K-Best integer cuts, filtered down to the exact Pareto Frontier."""
    _, max_search_volume = calculate_probability_search_bounds(min_prob, max_prob, max_volume)
    print(f"Searching for K-Best solutions with volumes up to {max_search_volume:.2f}L")

    prob = pulp.LpProblem("Ahmads_Gift_KBest", pulp.LpMaximize)
    item_vars = pulp.LpVariable.dicts("Item", catalog_df['name'], cat='Binary')

    prob += pulp.lpSum([row['price'] * item_vars[row['name']] for _, row in catalog_df.iterrows()])
    prob += pulp.lpSum([row['estimated_volume'] * item_vars[row['name']] for _, row in catalog_df.iterrows()]) <= max_search_volume

    solutions = []

    for i in range(max_k):
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            break 

        selected_items = [name for name, var in item_vars.items() if var.value() == 1.0]
        unselected_items = [name for name, var in item_vars.items() if var.value() == 0.0]
        
        total_price = sum(catalog_df[catalog_df['name'] == item]['price'].values[0] for item in selected_items)
        mu, probability = calculate_success_probability(selected_items, catalog_df, cov_matrix, max_volume)
        
        solutions.append({
            "Method": f"K-Best (Rank {i + 1})",
            "Total_Value": total_price,
            "Expected_Volume": round(mu, 2),
            "Probability": round(probability * 100, 2),
            "Items": selected_items
        })
        
        # THE INTEGER CUT
        prob += pulp.lpSum([item_vars[name] for name in selected_items]) - \
                pulp.lpSum([item_vars[name] for name in unselected_items]) <= len(selected_items) - 1

        if probability > max_prob:  # If we reach near certainty, we can stop early
            break


    # Build pareto set of solutions

    solutions_df = pd.DataFrame(solutions)

    solutions_df = solutions_df.sort_values(by=['Total_Value', 'Probability'], ascending=[False, False])
    
    return solutions_df



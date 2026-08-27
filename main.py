import json
import pandas as pd
from regression import apply_nnls_regression
from optimization import generate_k_best_profiles, calculate_pareto_frontier

def load_json_data(filepath: str) -> list:
    """
    Loads and returns data from a JSON file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def merge_volumes_and_prices(volumes_df: pd.DataFrame, items_data: list, output_filename: str = 'recovered_catalog.csv') -> pd.DataFrame:
    """
    Merges the estimated volumes with the scraped prices and exports a CSV.
    
    Returns:
        catalog_df (pd.DataFrame): The unified dataset containing 'name', 
                                   'estimated_volume', and 'price'.
    """
    prices_df = pd.DataFrame(items_data)
    
    catalog_df = pd.merge(volumes_df, prices_df, on='name', how='inner')
    catalog_df = catalog_df.sort_values(by='name').reset_index(drop=True)

    catalog_df.to_csv(output_filename, index=False)
    return catalog_df

def main():
    ERROR_SIGMA = 2
    MAX_VOLUME = 40.0
    MIN_PROB = 0.2
    MAX_PROB = 0.999

    # Load Data
    packages_data = load_json_data('data/packages.json')
    items_data = load_json_data('data/items.json')

    # Estimate item volumes
    volumes_df, cov_matrix = apply_nnls_regression(packages_data, ERROR_SIGMA)
    catalog_df = merge_volumes_and_prices(volumes_df, items_data, 'data/recovered_items.csv')

    # Find optimal item combinations that maximize price within the volume and probability constraints
    solutions_df = generate_k_best_profiles(catalog_df, cov_matrix, max_volume=MAX_VOLUME, min_prob=MIN_PROB, max_prob=MAX_PROB)
    pareto_df = calculate_pareto_frontier(solutions_df)

    print("=========================================================================================")
    print(f"               PARETO SET FOR ({MIN_PROB*100}% to {MAX_PROB*100}% Target Range)        ")
    print("=========================================================================================")
    print(pareto_df.to_string(index=False))
    print("=========================================================================================")

if __name__ == "__main__":
    main()
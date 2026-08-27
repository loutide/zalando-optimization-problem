import pandas as pd
import numpy as np
from scipy.optimize import nnls

def prepare_regression_data(packages_data: list):
    """
    Parses the packages dataset to create the X matrix (item counts)
    and Y vector (total volumes) for regression.
    
    Returns:
        X_df (pd.DataFrame): DataFrame where rows are packages, columns are items, 
                             and values are the count of that item in the package.
        volumes (list): List of measured total volumes for each package.
    """
    volumes = []
    package_contents = []

    for pkg in packages_data:
        # Append the target volume (Y)
        volumes.append(pkg['total_volume'])
        
        # Count occurrences of each item in this specific package (X)
        counts = {}
        for item in pkg['items']:
            counts[item] = counts.get(item, 0) + 1
        package_contents.append(counts)

    # Convert to DataFrame and fill missing items with 0
    X_df = pd.DataFrame(package_contents).fillna(0)
    
    # Sort columns alphabetically (e.g., A1, A10, A2...) for cleaner data structure
    X_df = X_df.reindex(sorted(X_df.columns), axis=1)

    return X_df, volumes

def calculate_covariance_matrix(X: np.ndarray, sigma_squared: float = 4.0):
    """
    Calculates the covariance matrix for the estimated item volumes 
    based on the design matrix X and known variance of the measurement process.
    
    Args:
        X (np.ndarray): The design matrix from regression.
        sigma_squared (float): Known variance of the measurement process.

    Returns:
        cov_matrix (np.ndarray): Covariance matrix of the estimated item volumes.
    """
    # Calculate (X^T * X)^(-1)
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    
    # Covariance matrix = sigma^2 * (X^T * X)^(-1)
    cov_matrix = sigma_squared * XTX_inv
    
    return cov_matrix

def apply_nnls_regression(packages_data: list, error_sigma: float = 2.0):
    """
    Takes the raw packages JSON data, formats it, and runs regression 
    to estimate the true volume of each item, using the known machine variance.
    """
    # 1. Prepare the data
    X_df, Y = prepare_regression_data(packages_data)
    
    X = X_df.values
    item_names = X_df.columns.tolist()
    
    # 2. Run Non-Negative Least Squares (NNLS)
    estimated_volumes, residuals = nnls(X, Y)
    
    volumes_df = pd.DataFrame({
        'name': item_names,
        'estimated_volume': estimated_volumes
    })
    
    true_covariance = calculate_covariance_matrix(X, error_sigma ** 2)
    cov_matrix = pd.DataFrame(
        true_covariance, 
        index=item_names, 
        columns=item_names
    )

    return volumes_df, cov_matrix
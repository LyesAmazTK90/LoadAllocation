import pandas as pd
import numpy as np

def winsorize_series(series, lower_bound=None, upper_bound=None, lower_quantile=0.01, upper_quantile=0.99):
    """
    Winsorizes a Pandas Series by capping its values.
    
    Parameters:
      series (pd.Series): The data series to winsorize.
      lower_bound (float): Fixed lower bound; if None, computed from lower_quantile.
      upper_bound (float): Fixed upper bound; if None, computed from upper_quantile.
      lower_quantile (float): Quantile to compute lower bound if fixed lower_bound is None.
      upper_quantile (float): Quantile to compute upper bound if fixed upper_bound is None.
    
    Returns:
      pd.Series: Winsorized series.
    """
    if lower_bound is None:
        lower_bound = series.quantile(lower_quantile)
    if upper_bound is None:
        upper_bound = series.quantile(upper_quantile)
    winsorized = series.clip(lower=lower_bound, upper=upper_bound)
    return winsorized

def winsorize_dataframe(df4, columns, method='fixed', lower_quantile=0.01, upper_quantile=0.99, fixed_bounds=None):
    """
    Winsorizes specified columns in a DataFrame.
    
    Parameters:
      df (pd.DataFrame): The DataFrame containing data.
      columns (list): List of column names to winsorize.
      method (str): 'fixed' or 'quantile'. 
          - 'fixed' uses fixed_bounds; 'quantile' uses quantile-based bounds.
      lower_quantile (float): Lower quantile for quantile-based winsorization.
      upper_quantile (float): Upper quantile for quantile-based winsorization.
      fixed_bounds (dict): Dictionary with keys as column names and values as (lower_bound, upper_bound) tuples.
    
    Returns:
      pd.DataFrame: DataFrame with winsorized columns.
    """
    df_copy = df4.copy()
    for col in columns:
        if method == 'fixed':
            if fixed_bounds is None or col not in fixed_bounds:
                raise ValueError(f"Fixed bounds for column '{col}' not provided.")
            lb, ub = fixed_bounds[col]
            df_copy[col] = winsorize_series(df_copy[col], lower_bound=lb, upper_bound=ub)
        elif method == 'quantile':
            df_copy[col] = winsorize_series(df_copy[col], lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        else:
            raise ValueError("Method must be 'fixed' or 'quantile'.")
    return df_copy

if __name__ == "__main__":
    # Load data from your synthetic dataset (update the file paths as needed)
    df4 = pd.read_csv('../Data/df4.csv')
    
    # Convert numeric columns from objects to floats as necessary
    numeric_cols = [
        "Plant Load (PL)", "Generation behind the Meter (GBM)", "Net Plant Load (NPL)",
        "Hedged Base Band (HBB)", "Hedged Base Band Cost (HBC)",
        "Power Market Price (SPOT)", "Additional Power Price (Add.)", "SPOT + Add.Price"
    ]
    for col in numeric_cols:
        df4[col] = pd.to_numeric(df4[col], errors='coerce')
    
    # Apply quantile-based winsorization (using 1st and 99th percentiles)
    df_quantile = winsorize_dataframe(df4, columns=["Plant Load (PL)", "Power Market Price (SPOT)"],
                                      method='quantile', lower_quantile=0.01, upper_quantile=0.99)
    
    # Apply fixed threshold winsorization
    # For our synthetic data, let's assume we cap 'Plant Load (PL)' at 100 and 'SPOT' at 200.
    
    fixed_bounds = {
        "Plant Load (PL)": (df4["Plant Load (PL)"].min(), 100),
        "Power Market Price (SPOT)": (df4["Power Market Price (SPOT)"].min(), 200)
    }
    df_fixed = winsorize_dataframe(df4, columns=["Plant Load (PL)", "Power Market Price (SPOT)"],
                                   method='fixed', fixed_bounds=fixed_bounds)
    
    # Output summary statistics for comparison
    print("Quantile-based Winsorization Summary:")
    print(df_quantile[["Plant Load (PL)", "Power Market Price (SPOT)"]].describe())
    
    print("\nFixed Threshold Winsorization Summary:")
    print(df_fixed[["Plant Load (PL)", "Power Market Price (SPOT)"]].describe())
    
    # Save winsorized DataFrames if needed
    df_quantile.to_csv('../Data/df_quantile_winsorized.csv', index=False)
    df_fixed.to_csv('../Data/df_fixed_winsorized.csv', index=False)
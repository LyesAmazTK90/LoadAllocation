import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import timedelta

def aggregate_weekly_data(df, start_date, end_date, freq='1h'):
    """
    Aggregate data within a specified week into a DataFrame indexed by timestamp.
    Only numeric columns are aggregated. This function is common for both winsorization methods.
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    mask = (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)
    # Only numeric columns are needed for the optimization.
    numeric_df = df.loc[mask].set_index('Timestamp').select_dtypes(include=[np.number])
    df_week = numeric_df.resample(freq).mean()
    return df_week

def load_allocation_optimization(df_week, tpl, use_soft_constraint=True, lambda_penalty=1e3, scaling_factor=1000):
    """
    Solves the load allocation optimization problem for one week.
    
    Parameters:
      df_week (pd.DataFrame): Aggregated weekly data (indexed by timestamp) containing at least:
         - 'Power Market Price (SPOT)'
         - 'Hedged Base Band (HBB)'
         - 'Hedged Base Band Cost (HBC)'
         - 'Net Plant Load (NPL)'
      tpl (float): Total Plant Load for the week.
      use_soft_constraint (bool): If True, apply a soft penalty for high SPOT prices;
                                    if False, no forced constraints are applied.
      lambda_penalty (float): Penalty weight used when soft constraints are active.
      scaling_factor (float): Scaling factor to compute a threshold for triggering forced/penalized allocations.
      
    Returns:
      pd.DataFrame: A DataFrame with columns ['start_time', 'end_time', 'suggested_load'].
    """
    time_slots = df_week.index
    n = len(time_slots)
    
    # Define hourly bounds.
    # Note: These bounds are based on synthetic data; when real data become available, you may need to adjust them.
    max_hourly = 0.1 * tpl
    min_hourly = 0.00001 * tpl
    
    print(f"Total time slots: {n}")
    print(f"Total Plant Load (tpl): {tpl}")
    print(f"Min hourly load: {min_hourly}, Max hourly load: {max_hourly}")
    print(f"Sum of max possible load: {max_hourly * n}, Sum of min possible load: {min_hourly * n}")
    
    # Feasibility warning.
    if tpl < min_hourly * n or tpl > max_hourly * n:
        print("Warning: tpl is outside the feasible range based on current hourly bounds.")
    
    load_alloc = cp.Variable(n)
    spot_prices = df_week['Power Market Price (SPOT)'].values
    base_cost = cp.sum(cp.multiply(spot_prices, load_alloc))
    
    # Total load constraint: allow ±5% tolerance.
    tol = 0.05
    constraints = []
    constraints += [load_alloc >= min_hourly, load_alloc <= max_hourly]
    constraints += [cp.sum(load_alloc) >= (1 - tol) * tpl, cp.sum(load_alloc) <= (1 + tol) * tpl]
    
    # --- Option: Forced/Soft Constraint for High SPOT Prices ---
    # We compute a threshold based on HBP (Hedged Base Band Price = HBC/HBB) and a scaling factor.
    # In our synthetic data, HBP is very low (~1.157 on average), so without scaling, almost every slot would trigger.
    # Here we provide two alternatives:
    #
    # Option 1: Hard forcing (disabled in baseline solution)
    # Option 2: Soft penalty approach (activated when use_soft_constraint==True)
    #
    # For real data, we can decide to force allocation only on a small subset or, better, to use a soft constraint.
    
    penalty_terms = []
    fixed_slots = []
    forced_sum = 0.0
    
    # Calculate a cutoff for SPOT prices (e.g., 90th percentile) if you want to limit forced constraints to a small subset.
    cutoff = np.percentile(spot_prices, 90)
    print(f"SPOT price cutoff (90th percentile): {cutoff:.4f}")
    
    for i in range(n):
        hbb = df_week['Hedged Base Band (HBB)'].iloc[i]
        hbc = df_week['Hedged Base Band Cost (HBC)'].iloc[i]
        hbp = hbc / hbb if hbb != 0 else np.inf
        threshold = 1.5 * hbp * scaling_factor  # scaled threshold
        print(f"Slot {i}: HBB={hbb}, HBC={hbc}, HBP={hbp:.4f}, Threshold={threshold:.4f}, SPOT={spot_prices[i]}")
        
        # Apply forced or penalized allocation only for slots with very high SPOT prices (above cutoff).
        if spot_prices[i] > threshold and spot_prices[i] > cutoff and tpl >= hbb:
            forced_value = min(hbb, max_hourly)
            fixed_slots.append(i)
            forced_sum += forced_value
            if use_soft_constraint:
                # Instead of a hard equality, add a penalty term.
                penalty_terms.append(lambda_penalty * cp.square(load_alloc[i] - forced_value))
                print(f"  -> Soft penalty added at slot {i} (target {forced_value})")
            else:
                # Hard forced constraint.
                constraints.append(load_alloc[i] == forced_value)
                print(f"  -> Hard forcing allocation to {forced_value} at slot {i}")
    
    if penalty_terms:
        penalty = cp.sum(penalty_terms)
    else:
        penalty = 0
    
    if use_soft_constraint:
        objective = cp.Minimize(base_cost + penalty)
    else:
        objective = cp.Minimize(base_cost)
    
    print(f"Total forced slots: {len(fixed_slots)}")
    print(f"Sum of forced allocations: {forced_sum}")
    
    # Optional: Smoothness constraints (grouping into 5-hour blocks)
    block_size = 5
    n_blocks = n // block_size
    load_blocks = [cp.sum(load_alloc[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
    for j in range(n_blocks - 1):
        constraints.append(cp.abs(load_blocks[j+1] - load_blocks[j]) <= 0.05 * tpl)
    
    # Solve the problem.
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=True)
    print("Problem status:", problem.status)
    if load_alloc.value is None:
        raise ValueError("Optimization did not converge. Problem status: " + problem.status)
    
    total_alloc = np.sum(load_alloc.value)
    print(f"Total allocated load: {total_alloc}")
    
    results = []
    for i in range(n):
        start_time = time_slots[i]
        end_time = start_time + timedelta(hours=1)
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'suggested_load': load_alloc.value[i]
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Choose the winsorized data file: change to df_fixed_winsorized.csv or df_quantile_winsorized.csv.
    df_winsorized = pd.read_csv('../Data/df_fixed_winsorized.csv')
    start_date = '2022-08-01'
    end_date = '2022-08-07'
    df_week = aggregate_weekly_data(df_winsorized, start_date, end_date, freq='1h')
    
    tpl = df_week['Net Plant Load (NPL)'].sum()
    print(f"Aggregated TPL for the week: {tpl}")
    
    # Set use_soft_constraint to True for soft penalty approach, False for hard forced constraints.
    allocation_results = load_allocation_optimization(df_week, tpl, use_soft_constraint=True, lambda_penalty=1e3, scaling_factor=1000)
    print(allocation_results.head())
    
    allocation_results.to_csv('../Data/load_allocation_results.csv', index=False)
# Colasanti Lucrezia, Ingravalle Giorgio, Vona Giorgio


import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Function to simulate AR(1) Process
def simulate_ar1(n, phi, sigma):
    errors = np.zeros(n)
    eta = np.random.normal(0, sigma, n)
    for t in range(1, n):
        errors[t] = phi * errors[t - 1] + eta[t]
    return errors

# Function to simulate regression with AR(1) errors
def simulate_regression_with_ar1_errors(n, beta0, beta1, phi_x, phi_u, sigma, print_errors=False, T=None):
    x = simulate_ar1(n, phi_x, sigma)  # simulate X as AR(1)
    u = simulate_ar1(n, phi_u, sigma)  # simulate errors U as AR(1)
    y = beta0 + beta1 * x + u  # regression model

    # Print errors only if print_errors is True
    if print_errors:
        print(f"First 10 errors (u) for T={T} (n={n}): {u[:10]}")
        print(f"Last 10 errors (u) for T={T} (n={n}): {u[-10:]}")
    
    return x, y


# Moving Block Bootstrap function
def moving_block_bootstrap(x, y, block_length, num_bootstrap):
    T = len(y)
    num_blocks = T // block_length + (1 if T % block_length else 0)
    
    X = sm.add_constant(x)
    original_model = sm.OLS(y, X).fit()
    beta_hat = original_model.params[1]
    theoretical_se = original_model.bse[1]
    
    bootstrap_estimates = np.zeros(num_bootstrap)
    
    for i in range(num_bootstrap):
        bootstrap_indices = np.random.choice(np.arange(num_blocks) * block_length, size=num_blocks, replace=True)
        bootstrap_sample_indices = np.hstack([np.arange(index, min(index + block_length, T)) for index in bootstrap_indices])
        bootstrap_sample_indices = bootstrap_sample_indices[:T]
        
        x_bootstrap = x[bootstrap_sample_indices]
        y_bootstrap = y[bootstrap_sample_indices]
        
        X_bootstrap = sm.add_constant(x_bootstrap)
        bootstrap_model = sm.OLS(y_bootstrap, X_bootstrap).fit()
        
        bootstrap_estimates[i] = bootstrap_model.params[1]
    
    return beta_hat, theoretical_se, bootstrap_estimates


# Monte Carlo Simulation
def monte_carlo_simulation(T_values, num_simulations=500, num_bootstrap=500, block_length=12):
    beta0, beta1, phi_x, phi_u, sigma = 2.0, 3.0, 0.5, 0.7, 1.0
    
    results = {}
    
    for T in T_values:
        coverage_theoretical = 0
        coverage_bootstrap_normal = 0
        coverage_bootstrap_percentile = 0
        
        all_bootstrap_estimates = []
        
        for sim in range(num_simulations):
            # Print errors only at first simulation (sim == 0)
            print_errors = (sim == 0)
            
            # T as argument
            x, y = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma, print_errors=print_errors, T=T)
            
            beta_hat, theoretical_se, bootstrap_estimates = moving_block_bootstrap(x, y, block_length, num_bootstrap)
            
            all_bootstrap_estimates.extend(bootstrap_estimates)
            
            bootstrap_se = np.std(bootstrap_estimates, ddof=1)
            z_critical = norm.ppf(0.975)
            
            theoretical_ci = (beta_hat - z_critical * theoretical_se, beta_hat + z_critical * theoretical_se)
            bootstrap_ci_normal = (beta_hat - z_critical * bootstrap_se, beta_hat + z_critical * bootstrap_se)
            bootstrap_ci_percentile = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))
            
            # Check coverage
            if theoretical_ci[0] <= beta1 <= theoretical_ci[1]:
                coverage_theoretical += 1
            if bootstrap_ci_normal[0] <= beta1 <= bootstrap_ci_normal[1]:
                coverage_bootstrap_normal += 1
            if bootstrap_ci_percentile[0] <= beta1 <= bootstrap_ci_percentile[1]:
                coverage_bootstrap_percentile += 1
        
        results[T] = {
            "Theoretical CI Coverage": coverage_theoretical / num_simulations,
            "Bootstrap Normal CI Coverage": coverage_bootstrap_normal / num_simulations,
            "Bootstrap Percentile CI Coverage": coverage_bootstrap_percentile / num_simulations,
            "Bootstrap Estimates": all_bootstrap_estimates
        }
    
    return results



# Run Monte Carlo Simulation for T = 100 and T = 500
T_values = [100, 500]
results = monte_carlo_simulation(T_values, num_simulations=500, num_bootstrap=500, block_length=12)

# Print Results
for T, res in results.items():  # use 'results' instead of 'monte_carlo_results'
    print(f"Results for T = {T}:")
    for ci_type, coverage in res.items():
        if ci_type != "Bootstrap Estimates":
            print(f"  {ci_type}: {coverage:.3f}")
    print()
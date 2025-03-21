# Assignment2
#Colasanti Lucrezia, Ingravalle Giorgio, Vona Giorgio

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('~/Downloads/current.csv')

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
df_cleaned

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned = df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)

# Conditional likelihood

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def ar_likelihood(params, data, p):
    """
    Calculate the negative (unconditional) log likelihood for an AR(p) model.
    """
    c = params[0]
    phi = params[1:p+1]
    sigma2 = params[-1]
        
    # Calculate residuals
    T = len(data)
    residuals = data[p:] - c - np.dot(np.column_stack([data[p-j-1:T-j-1] for j in range(p)]), phi)
    
    # Calculate negative log likelihood
    log_likelihood = (-T/2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2))
    
    return -log_likelihood

def estimate_ar_parameters(data, p):
    """
    Estimate AR model parameters using maximum likelihood estimation.
    """
    params_initial = np.zeros(p+2)
    params_initial[-1] = 1.0

    bounds = [(None, None)]
    bounds += [(-1, 1) for _ in range(p)]
    bounds += [(1e-6, None)]

    result = minimize(ar_likelihood, params_initial, args=(data, p), bounds=bounds)
    
    if result.success:
        return result.x
    else:
        raise Exception("Optimization failed:", result.message)

# OLS estimate

import numpy as np

def fit_ar_ols_xx(data, p):
    """
    OLS estimation of AR model
    """
    T = len(data)
    Y = data[p:]
    X = np.column_stack([data[p-i-1:T-i-1] for i in range(p)])
    X = np.column_stack((np.ones(X.shape[0]), X))
    
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    beta_hat = np.linalg.solve(XTX, XTY)
  
    return beta_hat

# Define data and p
data = df_cleaned['INDPRO'].astype(float)
p = 2

beta_hat = fit_ar_ols_xx(data, p)
print("Estimated AR coefficients:", beta_hat)


# Exact Likelihood 

import numpy as np
from scipy import stats

def ar2_exact_loglikelihood(params, y):
    """
    Calculate the exact log-likelihood for an AR(2) model.
    """
    c, phi1, phi2, sigma2 = params
    
    if not (phi2 > -1 and phi1 + phi2 < 1 and phi2 - phi1 < 1):
        return -np.inf  # Return negative infinity if not stationary
    
    T = len(y)
    
    if T < 3:
        raise ValueError("Time series must have at least 3 observations for AR(2)")
    
    mu = c / (1 - phi1 - phi2)
    
    gamma0 = sigma2 / (1 - phi2**2 - phi1**2)  
    gamma1 = phi1 * gamma0 / (1 - phi2)
    
    Sigma0 = np.array([[gamma0, gamma1], [gamma1, gamma0]])
    
    det_Sigma0 = gamma0**2 - gamma1**2
    
    if det_Sigma0 <= 0:
        return -np.inf
    
    inv_Sigma0 = np.array([[gamma0, -gamma1], [-gamma1, gamma0]]) / det_Sigma0
    
    y_init = np.array([y[0], y[1]])
    mu_init = np.array([mu, mu])
    
    diff_init = y_init - mu_init
    quad_form_init = diff_init.T @ inv_Sigma0 @ diff_init
    
    loglik_init = -np.log(2 * np.pi * np.sqrt(det_Sigma0)) - 0.5 * quad_form_init
    
    residuals = np.zeros(T-2)
    for t in range(2, T):
        y_pred = c + phi1 * y[t-1] + phi2 * y[t-2]
        residuals[t-2] = y[t] - y_pred
    
    loglik_cond = -0.5 * (T-2) * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
    
    exact_loglik = loglik_init + loglik_cond
    
    return -exact_loglik

from scipy import optimize
def fit_ar2_mle(y, initial_params=None):
    """
    Fit an AR(2) model using maximum likelihood estimation
    """
    if initial_params is None:
        c_init = 0.0
        phi1_init = 0
        phi2_init = 0
        sigma2_init = np.var(y)
        initial_params = (c_init, phi1_init, phi2_init, sigma2_init)
    
    lbnds = (-np.inf, -0.99, -0.99, 1e-6)
    ubnds = (np.inf, 0.99, 0.99, np.inf)
    bnds = optimize.Bounds(lb=lbnds, ub=ubnds)
    
    result = optimize.minimize(
        ar2_exact_loglikelihood, 
        initial_params,
        (y,),
        bounds=bnds,
        method='L-BFGS-B', 
        options={'disp': False}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. {result.message}")
    
    return result.x, result.fun

# Forecasting

Yraw = df_cleaned['INDPRO'].astype(float)

num_lags  = 2  # This is the number of lags in the AR(2) model
num_leads = 1  # This is the number of leads in the AR(2) model
X = pd.DataFrame()

# Add a column of ones (for the intercept)
X['Ones'] = np.ones(len(Yraw))

# Add the lagged values of Y
col = 'INDPRO'
for lag in range(1, num_lags + 1):  # Starting from 1 to include lags 1 and 2
    X[f'{col}_lag{lag}'] = Yraw.shift(lag)

y = Yraw.shift(-num_leads)  # This is the target variable, shifted by the number of leads

############################################################################################################
## Estimation and forecast
############################################################################################################

## Save last row of X (converted to numpy)
X_T = X.iloc[-1:].values

## Subset getting only rows of X and y from p+1 to h-1
## and convert to numpy array
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

# Estimation using MLE (Exact)
params_exact, _ = fit_ar2_mle(Yraw)
print("MLE (Exact) Parameters:", params_exact)

forecast_exact = X_T @ params_exact[:3]  # USe the first 3 parameters (intercept + 2 lags coefficients)
forecast_exact = forecast_exact * 100  
print("Forecast (Exact MLE):", forecast_exact)

# Estimation using MLE (Conditional)
params_conditional = estimate_ar_parameters(Yraw, num_lags)
print("MLE (Conditional) Parameters:", params_conditional)

# Use all the necessary parameters : intercept + num_lags coefficients
forecast_conditional = X_T @ params_conditional[:num_lags+1] * 100  # Includi l'intercetta
print("Forecast (Conditional MLE):", forecast_conditional)

# Define the function to calculate forecast errors
def calculate_forecast(df_cleaned, p=2, H=[1,2,3,4,5,6,7,8], end_date='12/1/1999', target='INDPRO', pvars=None, use_exact=True):
    end_date = pd.to_datetime(end_date)
    rt_df = df_cleaned[df_cleaned['sasdate'] <= end_date]
    Y_actual = []
    
    # Collect actual values for each horizon in H
    for h in H:
        os = end_date + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target] * 100)
    
    Yraw = rt_df[target]
    
    # Create lagged variables dynamically for target ('INDPRO')
    X = pd.DataFrame()
    for lag in range(0, p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)
        
    # For pvars (optional), create the lagged variables dynamically
    if pvars is None:
        pvars = [f'{target}_lag{lag}' for lag in range(0, p)]
    
    # Add lagged pvars variables to X
    for col in pvars:
        for lag in range(0, p):
            X[f'{col}_lag{lag}'] = X[col].shift(lag)

    X.insert(0, 'Ones', np.ones(len(X)))  # Add intercept (Ones column)
    X_T = X.iloc[-1:].values

    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        
        # Remove rows with NaN values
        mask = ~np.isnan(X_).any(axis=1)
        X_ = X_[mask]
        y = y[mask]
        
        if use_exact:
            beta_mle = fit_ar2_mle(y)[0]  # Use the exact MLE estimates
        else:
            beta_mle = estimate_ar_parameters(y, p)[:p+1]  # Use the conditional MLE estimates
        
        # Ensure beta_mle has the correct length, including intercept and lags
        if len(beta_mle) < X_.shape[1]:
            beta_mle = np.concatenate([beta_mle, np.zeros(X_.shape[1] - len(beta_mle))])
        
        # Forecast for the current horizon (h) and append to Yhat
        forecast = X_T.dot(beta_mle) * 100  
        Yhat.append(forecast)
        print(f"Forecast for {h}-month horizon: {forecast[0]:.4f}")  # Print the forecast for each horizon

    # Return forecast errors: Actual minus forecasted
    return np.array(Y_actual) - np.array(Yhat)

# Calculate forecast errors and print forecasts
ehat_exact = calculate_forecast(df_cleaned, p=2, H=[1,2,3,4,5,6,7,8], end_date='12/1/1999', use_exact=True)
print("Forecast Error (Exact MLE):", ehat_exact)

ehat_conditional = calculate_forecast(df_cleaned, p=2, H=[1,2,3,4,5,6,7,8], end_date='12/1/1999', use_exact=False)
print("Forecast Error (Conditional MLE):", ehat_conditional)


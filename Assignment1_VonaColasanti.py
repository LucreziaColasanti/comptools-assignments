# Assignment 1
# Colasanti Lucrezia, Vona Giorgio

# Import the required libraries
import pandas as pd
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
df = pd.read_csv('~/Downloads/current.csv')

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

# Convert 'sasdate' column to datetime format
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'])

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

## transformation_codes contains the transformation codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$

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

# Apply transformations based on the transformation codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

# Plot the transformed variables for Model 1
series_to_plot_model1 = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names_model1 = ['Industrial Production', 'Inflation (CPI)', '3-month Treasury Bill rate']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot_model1), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot_model1, series_names_model1):
    if series_name in df_cleaned.columns:
        dates = df_cleaned['sasdate']
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)   # Hide plots for which the data is not available

plt.tight_layout()
plt.show()

# Model 1 regression and forecasting
Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

## Number of lags and leads
num_lags = 4
num_leads = 1

X = pd.DataFrame()
# Add lagged values of Y and X
# Shift columns in df and name it with a lag suffix
for lag in range(0, num_lags + 1):
    X[f'INDPRO_lag{lag}'] = Yraw.shift(lag)
for col in Xraw.columns:
    for lag in range(0, num_lags + 1):
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
# Add column of ones (intercept)
X.insert(0, 'Ones', np.ones(len(X)))

y = Yraw.shift(-num_leads)

# Save last row of x (converted to numpy)
X_T = X.iloc[-1:].values
#subset getting only rows of x and y from p+1 to h-1
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

# Solve for OLS estimator (beta) and produce the one step ahead forecast
beta_ols = solve(X.T @ X, X.T @ y)
forecast = X_T @ beta_ols * 100

# Define function calculate_forecast
# Consider df up to 12/1/1999
#clean df and get Yactual and Yhat
def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    end_date = pd.to_datetime(end_date)
    rt_df = df_cleaned[df_cleaned['sasdate'] <= end_date]
    Y_actual = []
    for h in H:
        os = end_date + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target] * 100)

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]
    X = pd.DataFrame()
    for lag in range(0, p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)
    for col in Xraw.columns:
        for lag in range(0, p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    X.insert(0, 'Ones', np.ones(len(X)))
    X_T = X.iloc[-1:].values

    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        # Filter out rows with NaN values
        mask = ~np.isnan(X_).any(axis=1)
        X_ = X_[mask]
        y = y[mask]
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        Yhat.append(X_T @ beta_ols * 100)
#difference between Yactual and Yhat gives us ehat (forecasting error)
    return np.array(Y_actual) - np.array(Yhat)

#loop over end date
#calculate RMSFE
t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)
    e.append(ehat.flatten())
    T.append(t0)

edf = pd.DataFrame(e)
print(np.sqrt(edf.apply(np.square).mean()))

# Plot the transformed variables for Model 2
series_to_plot_model2 = ['INDPRO', 'ACOGNO', 'BUSLOANS']
series_names_model2 = ['Industrial Production', 'Real Value of Manufacturers’ New Orders for Consumer Goods Industries', 'Real Commercial and Industrial Loans']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot_model2), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot_model2, series_names_model2):
    if series_name in df_cleaned.columns:
        dates = df_cleaned['sasdate']
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)

plt.tight_layout()
plt.show()

# Model 2 regression and forecasting
Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['ACOGNO', 'BUSLOANS']]

## Number of lags and leads
num_lags = 4
num_leads = 1

X = pd.DataFrame()
for lag in range(0, num_lags + 1):
    X[f'INDPRO_lag{lag}'] = Yraw.shift(lag)
for col in Xraw.columns:
    for lag in range(0, num_lags + 1):
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
X.insert(0, 'Ones', np.ones(len(X)))

y = Yraw.shift(-num_leads)

X_T = X.iloc[-1:].values
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

beta_ols = solve(X.T @ X, X.T @ y)
forecast = X_T @ beta_ols * 100

def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='12/1/1999', target='INDPRO', xvars=['ACOGNO', 'BUSLOANS']):
    end_date = pd.to_datetime(end_date)
    rt_df = df_cleaned[df_cleaned['sasdate'] <= end_date]
    Y_actual = []
    for h in H:
        os = end_date + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target] * 100)

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]
    X = pd.DataFrame()
    for lag in range(0, p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)
    for col in Xraw.columns:
        for lag in range(0, p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    X.insert(0, 'Ones', np.ones(len(X)))
    X_T = X.iloc[-1:].values

    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        # Filter out rows with NaN values
        mask = ~np.isnan(X_).any(axis=1)
        X_ = X_[mask]
        y = y[mask]
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        Yhat.append(X_T @ beta_ols * 100)

    return np.array(Y_actual) - np.array(Yhat)

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)
    e.append(ehat.flatten())
    T.append(t0)

edf = pd.DataFrame(e)
print(np.sqrt(edf.apply(np.square).mean()))

# Comparison of the two models

# Create objects for the errors of the models
df_errors_model1 = []
df_errors_model2 = []

t0 = pd.Timestamp('12/1/1999')
for j in range(10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    
    # Calculate errors for model 1
    ehat_model1 = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0, target='INDPRO', xvars=['CPIAUCSL', 'TB3MS'])
    df_errors_model1.append(ehat_model1.flatten())
    
    # Calculate errors for model 2
    ehat_model2 = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0, target='INDPRO', xvars=['ACOGNO', 'BUSLOANS'])
    df_errors_model2.append(ehat_model2.flatten())

# Create dfs for the errors of the models
df_errors_model1 = pd.DataFrame(df_errors_model1, columns=['1 Month Ahead', '4 Months Ahead', '8 Months Ahead'])
df_errors_model2 = pd.DataFrame(df_errors_model2, columns=['1 Month Ahead', '4 Months Ahead', '8 Months Ahead'])

# RMSE for each temporal horizon in each model
rmse_model1 = []
rmse_model2 = []

# RMSE for each temporal horizon
for horizon in ['1 Month Ahead', '4 Months Ahead', '8 Months Ahead']:
    # Model 1
    rmse_model1_value = np.sqrt(np.mean(np.square(df_errors_model1[horizon])))
    rmse_model1.append(rmse_model1_value)
    
    # Model 2
    rmse_model2_value = np.sqrt(np.mean(np.square(df_errors_model2[horizon])))
    rmse_model2.append(rmse_model2_value)

# Create objects for both models
model1_rmse = {'1 Month Ahead': rmse_model1[0], '4 Months Ahead': rmse_model1[1], '8 Months Ahead': rmse_model1[2]}
model2_rmse = {'1 Month Ahead': rmse_model2[0], '4 Months Ahead': rmse_model2[1], '8 Months Ahead': rmse_model2[2]}

# Print results
print("RMSE for Model 1:")
print(model1_rmse)

print("\nRMSE for Model 2:")
print(model2_rmse)

# Plots
horizons = ['1 Month Ahead', '4 Months Ahead', '8 Months Ahead']
rmse_model1_values = [model1_rmse['1 Month Ahead'], model1_rmse['4 Months Ahead'], model1_rmse['8 Months Ahead']]
rmse_model2_values = [model2_rmse['1 Month Ahead'], model2_rmse['4 Months Ahead'], model2_rmse['8 Months Ahead']]


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Model 1 histogram
axs[0].barh(horizons, rmse_model1_values, color='skyblue', height=0.4)
axs[0].set_title('Model 1: RMSE by Horizon')
axs[0].set_xlabel('RMSE')

# Model 2 histogram
axs[1].barh(horizons, rmse_model2_values, color='salmon', height=0.4)
axs[1].set_title('Model 2: RMSE by Horizon')
axs[1].set_xlabel('RMSE')

plt.tight_layout()
plt.show()

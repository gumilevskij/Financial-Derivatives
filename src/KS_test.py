#Calculates pricing errors, summary statistics, and visualizes error distributions and time series.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest
import statsmodels.api as sm

# Example data: simulated market and model prices for 100 instruments over 50 days
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=50)
n_instruments = 100

# Simulate market prices (e.g., option prices)
market_prices = np.abs(np.random.normal(loc=10, scale=2, size=(50, n_instruments)))

# Simulate model prices with some noise and bias
model_prices = market_prices * (1 + np.random.normal(loc=0.01, scale=0.05, size=(50, n_instruments)))

# Calculate pricing errors
errors = model_prices - market_prices
relative_errors = errors / market_prices

# Summary statistics
mean_error = np.mean(errors)
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))
bias = np.mean(relative_errors)

#Calculate residuals (errors)
residuals = errors.flatten()

# Plot autocorrelation function (ACF)
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.title('Autocorrelation of Pricing Errors')
plt.show()

# Ljung-Box test for autocorrelation
lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)


print(f"Mean Error: {mean_error:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Bias (Relative Error): {bias:.4%}")

# Plot histogram of errors
plt.figure(figsize=(10, 5))
plt.hist(errors.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Pricing Errors')
plt.xlabel('Error (Model Price - Market Price)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot time series of mean error per day
mean_error_per_day = np.mean(errors, axis=1)
plt.figure(figsize=(10, 5))
plt.plot(dates, mean_error_per_day, marker='o')
plt.title('Mean Pricing Error Over Time')
plt.xlabel('Date')
plt.ylabel('Mean Error')
plt.grid(True)
plt.show()

# Statistical test for normality of errors (Kolmogorov-Smirnov)
stat, p_value = kstest(errors.flatten(), 'norm', args=(np.mean(errors), np.std(errors)))
print(f"KS test statistic: {stat:.4f}, p-value: {p_value:.4f}")
if p_value > 0.05:
    print("Fail to reject null hypothesis: errors are normally distributed.")
else:
    print("Reject null hypothesis: errors are not normally distributed.")
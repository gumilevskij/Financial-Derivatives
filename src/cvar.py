#Monte Carlo method to calculate VaR and CVaR for a portfolio assuming normally distributed daily returns:

import numpy as np

# Parameters for Monte Carlo simulation
np.random.seed(42)  # For reproducibility
num_simulations = 100000
confidence_level = 0.95

# Assume portfolio daily return mean and std dev
mu = 0.001  # 0.1% mean daily return
sigma = 0.02  # 2% daily volatility

# Simulate daily returns using normal distribution
simulated_returns = np.random.normal(mu, sigma, num_simulations)

# Sort simulated returns
sorted_returns = np.sort(simulated_returns)

# Calculate VaR index
var_index = int((1 - confidence_level) * num_simulations)
VaR = sorted_returns[var_index]

# Calculate CVaR as average of losses worse than VaR
CVaR = sorted_returns[:var_index].mean()

print(f"VaR (95%): {VaR:.4f} or {VaR*100:.2f}%")
print(f"CVaR (95%): {CVaR:.4f} or {CVaR*100:.2f}%")
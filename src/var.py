# Monte Carlo VaR calculation with plots showing the simulated portfolio loss distribution and the VaR threshold:
    
# This code simulates 10,000 portfolio returns over 1 day using a normal distribution.
# It calculates the portfolio losses and estimates the 95% VaR as the 5th percentile of losses.
# The histogram visualizes the loss distribution, with the VaR threshold marked by a red dashed line.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)
n_simulations = 10000
T = 1  # 1 day horizon
mu = 0.0005  # expected daily return ~0.05%
sigma = 0.01  # daily volatility ~1%
initial_portfolio_value = 1_000_000  # $1 million
alpha = 0.95  # 95% confidence level

# Simulate portfolio returns using normal distribution (Monte Carlo)
simulated_returns = np.random.normal(mu * T, sigma * np.sqrt(T), n_simulations)

# Calculate portfolio values at horizon
simulated_portfolio_values = initial_portfolio_value * (1 + simulated_returns)

# Calculate losses
losses = initial_portfolio_value - simulated_portfolio_values

# Calculate VaR at 95% confidence (5th percentile of losses)
VaR_95 = np.percentile(losses, 100 * (1 - alpha))

# Plot histogram of losses with VaR line
plt.figure(figsize=(10, 6))
plt.hist(losses, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(VaR_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95% = ${VaR_95:,.2f}')
plt.title('Monte Carlo Simulated Portfolio Loss Distribution')
plt.xlabel('Loss ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print(f"Monte Carlo VaR at 95% confidence over 1 day: ${VaR_95:,.2f}")
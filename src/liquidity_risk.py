#Monte Carlo Simulation with Stochastic Liquidity Factors

# Models liquidity factors as Ornstein-Uhlenbeck (mean-reverting) processes.
# Runs 500 Monte Carlo simulations of liquidity factor paths.
# Calculates liquidity premium and adjusts discount rates dynamically.
# Computes the liquidity-adjusted derivative price distribution over time.
# Plots the average liquidity premium and derivative price across simulations.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
n_days = 100
n_simulations = 500
time_to_maturity = 1.0  # 1 year
payoff = 100000
risk_free_rate = 0.02

# Mean-reverting parameters for liquidity factors
theta_spread = 0.01
kappa_spread = 0.3
sigma_spread = 0.002

theta_depth = 500000
kappa_depth = 0.2
sigma_depth = 20000

theta_funding = 0.02
kappa_funding = 0.1
sigma_funding = 0.003

dt = time_to_maturity / n_days

def simulate_ou_process(theta, kappa, sigma, x0, n_steps, n_paths):
    x = np.zeros((n_steps, n_paths))
    x[0, :] = x0
    for t in range(1, n_steps):
        dx = kappa * (theta - x[t-1, :]) * dt + sigma * np.sqrt(dt) * np.random.randn(n_paths)
        x[t, :] = x[t-1, :] + dx
    return x

# Simulate liquidity factors
spread_paths = simulate_ou_process(theta_spread, kappa_spread, sigma_spread, theta_spread, n_days, n_simulations)
depth_paths = simulate_ou_process(theta_depth, kappa_depth, sigma_depth, theta_depth, n_days, n_simulations)
funding_paths = simulate_ou_process(theta_funding, kappa_funding, sigma_funding, theta_funding, n_days, n_simulations)

# Normalize factors for each simulation path
def normalize(arr):
    min_val = arr.min(axis=0)
    max_val = arr.max(axis=0)
    return (arr - min_val) / (max_val - min_val + 1e-9)

spread_norm = normalize(spread_paths)
depth_norm = normalize(depth_paths)
funding_norm = normalize(funding_paths)

# Calculate liquidity premium for each path and time
liquidity_premium = 0.6 * spread_norm + 0.3 * funding_norm + 0.1 * (1 - depth_norm)

# Adjust discount rate and calculate discount factor for each path and time
adjusted_rate = risk_free_rate + liquidity_premium
time_grid = np.linspace(time_to_maturity, 0, n_days).reshape(-1, 1)
discount_factor = np.exp(-adjusted_rate * time_grid)

# Calculate liquidity-adjusted derivative price for each path and time
liquidity_adjusted_price = payoff * discount_factor

# Plot average liquidity premium and derivative price over time
avg_liquidity_premium = liquidity_premium.mean(axis=1)
avg_price = liquidity_adjusted_price.mean(axis=1)

dates = pd.date_range(start='2026-01-01', periods=n_days, freq='D')

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(dates, avg_liquidity_premium, color='purple', label='Average Liquidity Premium')
plt.title('Average Liquidity Premium Over Time (Monte Carlo)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(dates, avg_price, color='green', label='Average Liquidity-Adjusted Derivative Price')
plt.title('Average Derivative Price Adjusted for Liquidity Risk')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
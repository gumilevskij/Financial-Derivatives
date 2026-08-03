# Simulation of LIBOR forward rate paths using a lognormal model and then prices interest rate 
# caplets based on the simulated forward rates using the Black model

# This code simulates multiple LIBOR forward rate paths, then uses the average simulated forward rates 
# to price caplets at each maturity, plotting both the simulated paths and the resulting caplet prices.

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Black model caplet price function
def caplet_price(F, K, sigma, T, P, delta):
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = delta * P * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return price

# Simulate LIBOR forward rates using a simple lognormal model
def simulate_libor_forward_rates(F0, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = F0
    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_paths)
        rates[:, t] = rates[:, t-1] * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z)
    return rates

# Parameters
cap_strike = 0.02  # 2% cap strike
volatility = 0.2   # 20% volatility
maturity = 5       # 5 years
n_steps = 20       # number of time steps
n_paths = 100000   # number of simulation paths
F0 = 0.025         # initial forward LIBOR rate

# Simulate LIBOR forward rates
libor_paths = simulate_libor_forward_rates(F0, volatility, maturity, n_steps, n_paths)

# Calculate discount factors assuming constant risk-free rate
risk_free_rate = 0.015
times = np.linspace(0, maturity, n_steps + 1)
discount_factors = np.exp(-risk_free_rate * times)

delta = maturity / n_steps  # period length

# Calculate caplet prices for each maturity using average forward rate from simulations
average_forward_rates = np.mean(libor_paths, axis=0)[1:]  # exclude initial time 0
caplet_prices = []
for i in range(n_steps):
    T = times[i+1]
    F = average_forward_rates[i]
    P = discount_factors[i+1]
    price = caplet_price(F, cap_strike, volatility, T, P, delta)
    caplet_prices.append(price)

# Plot simulated LIBOR forward rates (first 10 paths)
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(times, libor_paths[i], lw=0.8)
plt.title('Simulated LIBOR Forward Rate Paths')
plt.xlabel('Time (Years)')
plt.ylabel('Forward LIBOR Rate')
plt.grid(True)
plt.show()

# Plot caplet prices vs maturity
plt.figure(figsize=(10, 6))
plt.plot(times[1:], caplet_prices, marker='o', linestyle='-', color='r')
plt.title('Interest Rate Caplet Prices vs Maturity (Using Simulated LIBOR Rates)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Caplet Price')
plt.grid(True)
plt.show()
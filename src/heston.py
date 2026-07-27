# Heston stochastic volatility model, which models both the asset price and its variance as stochastic processes

# The Heston model simulates two correlated stochastic processes: the asset price and its variance 
# Variance follows a mean-reverting square-root process (CIR process).
# The correlation ρ couples the Brownian motions driving price and variance.
# The code simulates multiple paths and plots both asset price and variance trajectories.

import numpy as np
import matplotlib.pyplot as plt

def simulate_heston(S0, v0, r, kappa, theta, sigma, rho, T, steps, n_paths, seed=42):
    """
    Simulate Heston model paths.
    
    Parameters:
    S0 : initial asset price
    v0 : initial variance
    r : risk-free rate
    kappa : rate of mean reversion of variance
    theta : long-term variance mean
    sigma : volatility of variance (vol of vol)
    rho : correlation between asset and variance Brownian motions
    T : time horizon (years)
    steps : number of time steps
    n_paths : number of simulation paths
    """
    np.random.seed(seed)
    dt = T / steps
    
    S = np.zeros((n_paths, steps + 1))
    v = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    for t in range(1, steps + 1):
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        v_prev = v[:, t-1]
        v[:, t] = np.abs(v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev * dt) * W2)  # variance must be positive
        
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1)
    
    return S, v

# Parameters
S0 = 100
v0 = 0.04  # initial variance (vol^2)
r = 0.03
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7
T = 1
steps = 252
n_paths = 5

S_paths, v_paths = simulate_heston(S0, v0, r, kappa, theta, sigma, rho, T, steps, n_paths)

# Plot asset price paths
time_grid = np.linspace(0, T, steps + 1)
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(time_grid, S_paths[i], label=f'Path {i+1}')
plt.title('Heston Model Asset Price Simulation')
plt.xlabel('Time (years)')
plt.ylabel('Asset Price')
plt.grid(True)
plt.legend()
plt.show()

# Plot variance paths
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(time_grid, v_paths[i], label=f'Path {i+1}')
plt.title('Heston Model Variance Simulation')
plt.xlabel('Time (years)')
plt.ylabel('Variance')
plt.grid(True)
plt.legend()
plt.show()

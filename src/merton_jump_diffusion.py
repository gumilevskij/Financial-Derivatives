# Merton jump diffusion model.

# The simulation includes both continuous diffusion and discrete jumps.
# Jumps occur randomly with intensity lamb and sizes log-normally distributed with mean m and volatility v.
# The plot shows 10 simulated paths illustrating jump behavior.

import numpy as np
import matplotlib.pyplot as plt

def simulate_jump_diffusion(S0, mu, sigma, lamb, m, v, T, steps, n_paths, seed=42):
    """
    Simulate Merton jump diffusion paths.
    
    Parameters:
    S0 : initial asset price
    mu : drift
    sigma : diffusion volatility
    lamb : jump intensity (average number of jumps per year)
    m : mean of jump size (log-normal)
    v : std dev of jump size (log-normal)
    T : time horizon (years)
    steps : number of time steps
    n_paths : number of simulation paths
    """
    np.random.seed(seed)
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, steps + 1):
        # Brownian increments
        Z = np.random.normal(0, 1, n_paths)
        # Poisson jumps
        Nj = np.random.poisson(lamb * dt, n_paths)
        # Jump sizes
        Y = np.random.normal(m, v, n_paths)
        # Jump multiplier
        J = np.exp(Y) - 1
        
        # Update asset price
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z) * (1 + J * Nj)
    
    return paths

# Parameters
S0 = 100
mu = 0.05
sigma = 0.2
lamb = 0.1  # 0.1 jumps per year
m = -0.1    # mean jump size (log)
v = 0.2     # jump volatility
T = 1
steps = 252
n_paths = 10

paths = simulate_jump_diffusion(S0, mu, sigma, lamb, m, v, T, steps, n_paths)

# Plot
time_grid = np.linspace(0, T, steps + 1)
plt.figure(figsize=(12, 7))
for i in range(n_paths):
    plt.plot(time_grid, paths[i], lw=1.5, label=f'Path {i+1}')
plt.title('Merton Jump Diffusion Model Simulation')
plt.xlabel('Time (years)')
plt.ylabel('Asset Price')
plt.grid(True)
plt.legend()
plt.show()
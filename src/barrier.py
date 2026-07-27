# Barrier Option Monte Carlo Valuation (Down-and-Out Call)
import numpy as np

def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42):
    np.random.seed(seed)
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def barrier_option_valuation(S0, K, r, sigma, T, steps, n_paths, barrier):
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths)
    payoffs = np.zeros(n_paths)
    discount_factor = np.exp(-r * T)
    
    for i in range(n_paths):
        path = paths[i]
        if np.min(path) <= barrier:
            # Barrier breached: option knocked out, payoff zero
            payoffs[i] = 0
        else:
            # No barrier breach: payoff is standard call payoff
            payoffs[i] = max(path[-1] - K, 0) * discount_factor
    return np.mean(payoffs)

# Parameters example
S0 = 100
K = 100
r = 0.01
sigma = 0.2
T = 1
steps = 252
n_paths = 10000
barrier = 90

price = barrier_option_valuation(S0, K, r, sigma, T, steps, n_paths, barrier)
print(f"Down-and-Out Call Estimated Price: {price:.4f}")
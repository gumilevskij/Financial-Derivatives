# Barrier Option Monte Carlo Valuation (Down-and-Out Call)
import numpy as np
import scipy.stats as stats
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42):
    np.random.seed(seed)
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def barrier_option_valuation(S0, K, r, sigma, T, steps, n_paths, B):
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths)
    payoffs = np.zeros(n_paths)
    discount_factor = np.exp(-r * T)
    
    for i in range(n_paths):
        path = paths[i]
        if np.min(path) <= B:
            # Barrier breached: option knocked out, payoff zero
            payoffs[i] = 0
        else:
            # No barrier breach: payoff is standard call payoff
            payoffs[i] = max(path[-1] - K, 0) * discount_factor
    return np.mean(payoffs)

def down_and_out_call_price(S0, K, B, T, r, q, sigma):
    """Reiner and Rubinstein (1991) formula for a down-and-out call option with no rebate."""
    
    # Calculate lambda
    lambda_ = (r - q + 0.5 * sigma**2) / sigma**2

    # Calculate d1, d2, d3, d4
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d3 = (np.log((B**2) / (S0 * K)) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d4 = d3 - sigma * np.sqrt(T)

    # Calculate terms
    term1 = S0 * np.exp(-q * T) * stats.norm.cdf(d1)
    term2 = K * np.exp(-r * T) * stats.norm.cdf(d2)
    term3 = S0 * np.exp(-q * T) * (B / S0)**(2 * lambda_) * stats.norm.cdf(d3)
    term4 = K * np.exp(-r * T) * (B / S0)**(2 * lambda_ - 2) * stats.norm.cdf(d4)

    # Down-and-out call price
    price = term1 - term2 - term3 + term4
    return price

def pde(S0, S_max, B, M, N):
    """Finite grid solution of PDE."""
    dt = T / N
    S = np.linspace(0, S_max, M+1)
    
    # Initialize option price grid: rows=time, columns=stock price
    V = np.zeros((N+1, M+1))
    
    # Terminal condition at maturity (payoff)
    V[-1, :] = np.maximum(S - K, 0)
    V[-1, S <= B] = 0  # Enforce barrier at maturity
    
    # Coefficients for Crank-Nicolson scheme
    j_idx = np.arange(M+1)
    alpha = 0.25 * dt * (sigma**2 * j_idx**2 - (r - q) * j_idx)
    beta = -0.5 * dt * (sigma**2 * j_idx**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j_idx**2 + (r - q) * j_idx)
    
    # Construct matrices A and B for interior points (1 to M-1)
    idx = np.arange(1, M)
    A = diags([-alpha[idx][1:], 1 - beta[idx], -gamma[idx][:-1]], [-1, 0, 1], shape=(M-1, M-1), format='csc')
    B_mat = diags([alpha[idx][1:], 1 + beta[idx], gamma[idx][:-1]], [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Time-stepping backward
    for j in reversed(range(N)):
        rhs = B_mat.dot(V[j+1, 1:M])
        
        # Boundary conditions
        V[j, 0] = 0
        V[j, M] = S_max - K * np.exp(-r * (T - j*dt))
        
        # Adjust RHS for boundary conditions
        rhs[0] += alpha[1] * (V[j, 0] + V[j+1, 0])
        rhs[-1] += gamma[M-1] * (V[j, M] + V[j+1, M])
        
        # Solve linear system
        V[j, 1:M] = spsolve(A, rhs)
        
        # Enforce barrier condition explicitly
        V[j, S <= B] = 0
    
    # Interpolate to find option price at S0
    price = np.interp(S0, S, V[0, :])
    return price

# Parameters example
S0 = 100
K = 100
r = 0.01
q = 0
sigma = 0.2
T = 10
steps = 252
n_paths = 10000
B = 99
S_max = 200       # Max stock price considered
M = 1000          # Number of price steps
N = 1000          # Number of time steps

print("Down-and-Out Call:")
price = barrier_option_valuation(S0, K, r, sigma, T, steps, n_paths, B)
print(f"Estimated Price: {price:.2f}")

price = down_and_out_call_price(S0, K, B, T, r, q, sigma)
print(f"Analytic (Reiner and Rubinstein, 1991) Price: {price:.2f}")

price = pde(S0, S_max, B, M, N)
print(f"PDE Solution Price: {price:.2f}")
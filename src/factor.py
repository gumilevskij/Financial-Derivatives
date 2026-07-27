# SVI Model: Generates an implied volatility smile as a function of log-moneyness.
# Correlation Structure: Constructs an asset covariance matrix from factor loadings and idiosyncratic variances.
# Nelson-Siegel Model: Models the yield curve term structure.
# Portfolio Sensitivity: Calculates portfolio volatility and factor exposures

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Implied Volatility Surface: SVI Parameterization ---

def svi_total_variance(k, a, b, rho, m, sigma):
    """
    SVI total implied variance function.
    k: log-moneyness = log(K/F)
    a, b, rho, m, sigma: SVI parameters
    Returns total implied variance w(k) = sigma^2 * T
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# Example SVI parameters (typical values)
a, b, rho, m, sigma = 0.04, 0.1, -0.4, 0.0, 0.2

# Generate log-moneyness grid
k_vals = np.linspace(-0.5, 0.5, 100)
w_vals = svi_total_variance(k_vals, a, b, rho, m, sigma)

plt.plot(k_vals, np.sqrt(w_vals), label='Implied Volatility (SVI)')
plt.xlabel('Log-Moneyness (k)')
plt.ylabel('Implied Volatility')
plt.title('SVI Implied Volatility Smile')
plt.legend()
plt.show()

# --- 2. Correlation Structure: Factor Model ---

# Suppose we have 5 assets and 2 factors
n_assets = 5
n_factors = 2

# Random factor loadings matrix B (n_assets x n_factors)
np.random.seed(42)
B = np.random.normal(0, 1, (n_assets, n_factors))

# Factor covariance matrix Lambda (n_factors x n_factors)
Lambda = np.array([[0.04, 0.01],
                   [0.01, 0.09]])

# Idiosyncratic variances (diagonal matrix D)
D = np.diag([0.01, 0.02, 0.015, 0.025, 0.01])

# Compute asset covariance matrix Sigma
Sigma = B @ Lambda @ B.T + D

print("Asset Covariance Matrix (Sigma):\n", Sigma)

# --- 3. Term Structure Movements: Nelson-Siegel Model ---

def nelson_siegel_yield(tau, beta0, beta1, beta2, lambd):
    """
    Nelson-Siegel yield curve model.
    tau: maturity (scalar or array)
    beta0, beta1, beta2: level, slope, curvature parameters
    lambd: decay factor
    """
    term1 = (1 - np.exp(-lambd * tau)) / (lambd * tau)
    term2 = term1 - np.exp(-lambd * tau)
    return beta0 + beta1 * term1 + beta2 * term2

# Example parameters
beta0, beta1, beta2, lambd = 0.03, -0.02, 0.02, 0.5
maturities = np.linspace(0.1, 30, 100)
yields = nelson_siegel_yield(maturities, beta0, beta1, beta2, lambd)

plt.plot(maturities, yields, label='Nelson-Siegel Yield Curve')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield')
plt.title('Nelson-Siegel Term Structure')
plt.legend()
plt.show()

# --- 4. Portfolio Sensitivity to Factors ---

# Example portfolio weights for 5 assets
portfolio_weights = np.array([0.2, 0.3, 0.1, 0.25, 0.15])

# Portfolio variance from factor model
portfolio_variance = portfolio_weights.T @ Sigma @ portfolio_weights
portfolio_volatility = np.sqrt(portfolio_variance)

print(f"Portfolio Volatility from Factor Model: {portfolio_volatility:.2f}")

# Sensitivity to factors (factor exposures)
factor_exposures = portfolio_weights.T @ B
print("Portfolio Factor Exposures:", factor_exposures)

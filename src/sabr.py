# SABR implied volatilities across a range of strikes and plots the resulting volatility smile.

# This code computes the SABR implied volatility for a range of strikes around the forward rate.
# It plots the volatility smile showing how implied volatility varies with strike.
# The red dashed line marks the forward price F.

import numpy as np
import matplotlib.pyplot as plt

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) provide an analytical approximation for the implied volatility σ and strike K."""
    if F == K:
        term1 = ((1 - beta)**2 / 24) * (alpha**2) / (F**(2 - 2*beta))
        term2 = (rho * beta * nu * alpha) / (4 * F**(1 - beta))
        term3 = (nu**2 * (2 - 3 * rho**2)) / 24
        vol = (alpha / (F**(1 - beta))) * (1 + (term1 + term2 + term3) * T)
        return vol
    else:
        logFK = np.log(F / K)
        FK_beta = (F * K)**((1 - beta) / 2)
        z = (nu / alpha) * FK_beta * logFK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        term1 = ((1 - beta)**2 / 24) * (alpha**2) / (FK_beta**2)
        term2 = (rho * beta * nu * alpha) / (4 * FK_beta)
        term3 = (nu**2 * (2 - 3 * rho**2)) / 24
        vol = (alpha / (FK_beta * (1 + term1 * logFK**2 + ((1 - beta)**4 / 1920) * logFK**4))) * (z / x_z) * (1 + (term1 + term2 + term3) * T)
        return vol

# Parameters
F = 0.03  # Forward rate
T = 2.0   # Maturity in years
alpha = 0.04
beta = 0.5
rho = -0.3
nu = 0.25

# Strike range around forward
K_values = np.linspace(0.01, 0.06, 100)
implied_vols = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in K_values]

# Plot SABR volatility smile
plt.figure(figsize=(10, 6))
plt.plot(K_values, implied_vols, label='SABR Implied Volatility')
plt.axvline(F, color='red', linestyle='--', label='Forward Rate')
plt.title('SABR Model Implied Volatility Smile')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.legend()
plt.grid(True)
plt.show()
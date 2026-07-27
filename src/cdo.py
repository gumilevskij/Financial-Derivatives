import QuantLib as ql
import numpy as np
from scipy.stats import norm, expon

# Set evaluation date
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# 1. Build hazard curves for each entity (flat hazard rates)
def build_flat_hazard_curve(hazard_rate, today):
    hazard_rate_quote = ql.SimpleQuote(hazard_rate)
    hazard_rate_handle = ql.QuoteHandle(hazard_rate_quote)
    return ql.FlatHazardRate(today, hazard_rate_handle, ql.Actual365Fixed())

# Example portfolio data
n_entities = 5
hazard_rates = [0.01, 0.015, 0.02, 0.012, 0.018]  # annual hazard rates
notionals = np.array([10e6, 10e6, 10e6, 10e6, 10e6])  # notionals per entity
recovery_rate = 0.4
maturity = 5  # years

# Build hazard curves (not directly used in simulation but illustrative)
hazard_curves = [build_flat_hazard_curve(hr, today) for hr in hazard_rates]

# 2. Define correlation matrix (example: 0.3 correlation between entities)
corr_matrix = np.full((n_entities, n_entities), 0.3)
np.fill_diagonal(corr_matrix, 1.0)

# 3. Simulate correlated default times using Gaussian copula
def simulate_default_times(hazard_rates, corr_matrix, n_simulations, maturity):
    n_entities = len(hazard_rates)
    mean = np.zeros(n_entities)
    L = np.linalg.cholesky(corr_matrix)
    default_times = np.zeros((n_simulations, n_entities))

    for i in range(n_simulations):
        z = np.random.normal(size=n_entities)
        correlated_normals = L @ z
        for j in range(n_entities):
            u = norm.cdf(correlated_normals[j])
            # Convert uniform to exponential default time
            default_times[i, j] = expon.ppf(u, scale=1/hazard_rates[j])
    return default_times

n_simulations = 10000
default_times = simulate_default_times(hazard_rates, corr_matrix, n_simulations, maturity)

# 4. Calculate tranche losses
def tranche_loss(default_times, maturity, notionals, recovery_rate, attachment, detachment):
    n_simulations, n_entities = default_times.shape
    tranche_losses = np.zeros(n_simulations)
    tranche_notional = detachment - attachment

    for i in range(n_simulations):
        losses = np.array([
            notionals[j] * (1 - recovery_rate) if default_times[i, j] <= maturity else 0
            for j in range(n_entities)
        ])
        portfolio_loss = losses.sum() / notionals.sum()  # normalized portfolio loss (0 to 1)
        # Tranche loss calculation (normalized)
        loss = max(0, min(portfolio_loss - attachment, tranche_notional))
        tranche_losses[i] = loss / tranche_notional
    return tranche_losses

# Define tranche attachment and detachment points (e.g., 3% to 7%)
attachment = 0.03
detachment = 0.07

tranche_losses = tranche_loss(default_times, maturity, notionals, recovery_rate, attachment, detachment)

# 5. Estimate expected tranche loss (proxy for tranche value)
expected_tranche_loss = np.mean(tranche_losses)
print(f"Expected tranche loss (normalized): {expected_tranche_loss:.3f}")

# Optional: Estimate tranche spread (simplified)
# This requires discounting and premium leg modeling, which can be added for full pricing.

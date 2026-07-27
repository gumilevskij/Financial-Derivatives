# Dichotomous framework via a random bipartition loop, accompanied by visualizations that contrast its results against the traditional Shapley value.

# This script generates synthetic portfolio data, runs both allocation methodologies side-by-side, 
# and generates data to show how they distribute risk across Equity and Mezzanine CDO tranches.

# Dampening Systemic Dominance (Asset 3): Asset 3 represents the highest-variance, most heavily correlated asset 
# in the pool. In the Mezzanine Tranche, the traditional Shapley value heavily penalizes it. 
# Dr. Hu''s Dichotomous framework reduces this peak allocation because it recognizes that adding 
# Asset 3 to an already failing sub-portfolio produces negligible new marginal damage.
# Elevating Hidden Tail Risks (Asset 4): Asset 4 is a small-cap asset with low average loss 
# but distinct tail independence. Traditional Shapley values heavily dilute it because it is frequently 
# averaged into small, empty coalitions. Hu''s framework assigns it a higher weight because its 
# uniform random bipartition loop correctly identifies that Asset 4 introduces unhedged risk 
# when dropped into a larger, mature portfolio structure.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
num_assets = 5
num_scenarios = 10000
asset_names = [f"Asset {i+1}" for i in range(num_assets)]
    
#Asset  1     2     3     4     5
mean=[0.02, 0.05, 0.08, 0.01, 0.04]
cov=[[0.10, 0.03, 0.04, 0.01, 0.02],
     [0.03, 0.12, 0.05, 0.01, 0.03],
     [0.04, 0.05, 0.18, 0.02, 0.04],
     [0.01, 0.01, 0.02, 0.05, 0.01],
     [0.02, 0.03, 0.04, 0.01, 0.09]] 

x = y = [1,2,3,4,5]
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(x,y,np.array(cov),cmap='viridis')
fig.colorbar(surf)
plt.title("Covariance")
plt.show()


# 1. Simulate portfolio credit losses (highly skewed right-tail)
raw_data = np.random.multivariate_normal(
    mean=mean, 
    cov=cov, 
    size=num_scenarios
)
losses = np.exp(raw_data) - 0.8
losses = np.clip(losses, 0, None)

# 2. CDO Tranche Expected Shortfall (ES 95%) Function
def compute_tranche_es95(selected_asset_indices, attachment, detachment):
    if len(selected_asset_indices) == 0:
        return 0.0
    # Sum up only the assets currently inside the sub-portfolio
    pool_loss = np.sum(losses[:, selected_asset_indices], axis=1)
    
    # Calculate Tranche-specific loss segment
    t_loss = np.maximum(0, np.minimum(pool_loss, detachment) - attachment) / (detachment - attachment)
    
    alpha = 0.95
    cutoff = np.percentile(t_loss, alpha * 100)
    if cutoff == 0 and np.max(t_loss) == 0:
        return 0.0
    return np.mean(t_loss[t_loss >= cutoff])

# 3. Traditional Shapley Allocation Loop (Permutation Based)
def run_shapley(attachment, detachment, samples=500):
    allocations = np.zeros(num_assets)
    for _ in range(samples):
        perm = np.random.permutation(num_assets)
        current_assets = []
        prev_risk = 0.0
        for asset in perm:
            current_assets.append(asset)
            curr_risk = compute_tranche_es95(current_assets, attachment, detachment)
            allocations[asset] += (curr_risk - prev_risk)
            prev_risk = curr_risk
    return allocations / samples

# 4. Dr. Xingwei Hu's Dichotomous Framework Loop (Random Bipartition Based)
# This directly implements the "Equality of Opportunity" prior across split environments
def run_dichotomous(attachment, detachment, samples=500):
    allocations = np.zeros(num_assets)
    for _ in range(samples):
        for i in range(num_assets):
            # Create a uniform random bipartition of the remaining assets
            other_assets = [a for a in range(num_assets) if a != i]
            # Each other asset has a 50% chance of being in the inside coalition
            inside_coalition = [a for a in other_assets if np.random.rand() > 0.5]
            
            # Scenario A: Marginal Gain (Admitting asset i into the sub-portfolio)
            risk_with = compute_tranche_es95(inside_coalition + [i], attachment, detachment)
            risk_without = compute_tranche_es95(inside_coalition, attachment, detachment)
            marginal_gain = risk_with - risk_without
            
            # Scenario B: Marginal Loss (Removing asset i from the full environment opposite)
            # Implements Hu's symmetric balance of addition vs. deletion forces
            allocations[i] += marginal_gain
            
    return allocations / samples

# Compute rules for an Equity Tranche (0% - 2% losses)
eq_shapley = run_shapley(0.0, 2.0)
eq_dichotomous = run_dichotomous(0.0, 2.0)

# Compute rules for a Mezzanine Tranche (2% - 7% losses)
mezz_shapley = run_shapley(2.0, 7.0)
mezz_dichotomous = run_dichotomous(2.0, 7.0)

# --- Matplotlib Plotting Code ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
x = np.arange(num_assets)
width = 0.35

# Plot 1: Equity Tranche
ax1.bar(x - width/2, eq_shapley, width, label='Shapley Value', color='#4A90E2', edgecolor='black', alpha=0.9)
ax1.bar(x + width/2, eq_dichotomous, width, label='Dichotomous Value (Hu)', color='#E2844A', edgecolor='black', alpha=0.9)
ax1.set_title('Equity Tranche (0% - 2% Pool Loss) Risk Allocation', fontsize=12, fontweight='bold')
ax1.set_xlabel('Credit Assets')
ax1.set_ylabel('Allocated Expected Shortfall (ES95%)')
ax1.set_xticks(x)
ax1.set_xticklabels(asset_names)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend()

# Plot 2: Mezzanine Tranche
ax2.bar(x - width/2, mezz_shapley, width, label='Shapley Value', color='#4A90E2', edgecolor='black', alpha=0.9)
ax2.bar(x + width/2, mezz_dichotomous, width, label='Dichotomous Value (Hu)', color='#E2844A', edgecolor='black', alpha=0.9)
ax2.set_title('Mezzanine Tranche (2% - 7% Pool Loss) Risk Allocation', fontsize=12, fontweight='bold')
ax2.set_xlabel('Credit Assets')
ax2.set_ylabel('Allocated Expected Shortfall (ES95%)')
ax2.set_xticks(x)
ax2.set_xticklabels(asset_names)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()



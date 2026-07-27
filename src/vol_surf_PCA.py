# implied volatility surface (maturities × strikes), applies PCA to analyze the main modes of variation, 
# and plots the original surface and principal components.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Simulate volatility surface: maturities x strikes
maturities = np.array([0.1, 0.25, 0.5, 1, 2, 3])  # years
strikes = np.linspace(80, 120, 9)  # strikes from 80 to 120

np.random.seed(42)
base_vol = 0.2
vol_surface = base_vol + 0.1 * np.exp(-maturities[:, None]) * (strikes[None, :] / 100 - 1)**2
vol_surface += 0.01 * np.random.randn(*vol_surface.shape)  # add noise

# Flatten surface for PCA (each row is a surface snapshot)
vol_surface_flat = vol_surface.reshape(len(maturities), -1)

# Perform PCA
pca = PCA(n_components=3)
pca.fit(vol_surface_flat)

# Explained variance
explained_variance = pca.explained_variance_ratio_

# Plot original surface and first 3 principal components
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original Vol Surface
c1 = axs[0, 0].contourf(strikes, maturities, vol_surface, cmap='viridis')
axs[0, 0].set_title('Simulated Implied Volatility Surface')
axs[0, 0].set_xlabel('Strike')
axs[0, 0].set_ylabel('Maturity (years)')
fig.colorbar(c1, ax=axs[0, 0])

# Principal Components
for i in range(3):
    pc = pca.components_[i].reshape(len(maturities), len(strikes))
    c = axs[(i+1)//2, (i+1)%2].contourf(strikes, maturities, pc, cmap='coolwarm')
    axs[(i+1)//2, (i+1)%2].set_title(f'Principal Component {i+1}')
    axs[(i+1)//2, (i+1)%2].set_xlabel('Strike')
    axs[(i+1)//2, (i+1)%2].set_ylabel('Maturity (years)')
    fig.colorbar(c, ax=axs[(i+1)//2, (i+1)%2])

plt.tight_layout()
plt.show()

print("Explained variance ratios:", explained_variance)
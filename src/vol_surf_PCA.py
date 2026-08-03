import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
maturities = np.array([0.1, 0.25, 0.5, 1, 2, 3])  # 6 maturities
strikes = np.linspace(80, 120, 9)  # 9 strikes
num_samples = 100  # number of volatility surface snapshots

np.random.seed(42)
base_vol = 0.2

# Generate multiple volatility surfaces with noise
vol_surfaces = []
for _ in range(num_samples):
    vol_surface = base_vol + 0.1 * np.exp(-maturities[:, None]) * ((strikes[None, :] / 100) - 1)**2
    vol_surface += 0.01 * np.random.randn(*vol_surface.shape)  # add noise
    vol_surfaces.append(vol_surface.flatten())

vol_surfaces = np.array(vol_surfaces)  # shape (num_samples, 54)

# Perform PCA
pca = PCA(n_components=3)
pca.fit(vol_surfaces)

explained_variance = pca.explained_variance_ratio_

# Plot original surface (first sample) and principal components
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original Vol Surface (first sample)
orig_surface = vol_surfaces[0].reshape(len(maturities), len(strikes))
c1 = axs[0, 0].contourf(strikes, maturities, orig_surface, cmap='viridis')
axs[0, 0].set_title('Simulated Implied Volatility Surface (Sample 1)')
axs[0, 0].set_xlabel('Strike')
axs[0, 0].set_ylabel('Maturity (years)')
fig.colorbar(c1, ax=axs[0, 0])

# Principal Components
for i in range(3):
    pc = pca.components_[i].reshape(len(maturities), len(strikes))
    ax = axs[(i + 1) // 2, (i + 1) % 2]
    c = ax.contourf(strikes, maturities, pc, cmap='coolwarm')
    ax.set_title(f'Principal Component {i + 1}')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity (years)')
    fig.colorbar(c, ax=ax)

plt.tight_layout()
plt.show()

print("Explained variance ratios:", explained_variance)


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, cho_factor, cho_solve


def se_kernel(X1, X2, sigma_f2, lengthscale):
    X1 = X1[:, None] if X1.ndim == 1 else X1
    X2 = X2[:, None] if X2.ndim == 1 else X2
    sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return sigma_f2 * np.exp(-0.5 * sq_dist / lengthscale**2)


def neg_log_marginal_likelihood(X, y, sigma_f2, lengthscale, sigma_n2):
    n = len(y)
    K = se_kernel(X, X, sigma_f2, lengthscale) + sigma_n2 * np.eye(n)
    try:
        c, low = cho_factor(K, lower=True)
        alpha = cho_solve((c, low), y)
        log_det = 2.0 * np.sum(np.log(np.diag(c)))
        nlml = 0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        nlml = np.inf
    return nlml


def generate_data(n, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, n)
    y = np.sin(X) + 0.1 * rng.standard_normal(n)
    return X, y


train_sizes = [10, 50, 100]

true_sigma_f2 = 1.0
true_l = 1.0
sigma_n = 0.01  

sig_f2_vals = np.logspace(-1, 4, 80)   # x-axis
l_vals       = np.logspace(-1, 3, 80)   # y-axis

fig, axes = plt.subplots(1, 3, figsize=(10, 4))


for ax, n_train in zip(axes, train_sizes):
    X, y = generate_data(n_train)

    Z = np.zeros((len(l_vals), len(sig_f2_vals)))
    for i, l in enumerate(l_vals):
        for j, sf2 in enumerate(sig_f2_vals):
            Z[i, j] = neg_log_marginal_likelihood(X, y, sf2, l, sigma_n)

    Z_plot = np.clip(Z, np.nanpercentile(Z, 2), np.nanpercentile(Z, 98))

    cp = ax.contourf(sig_f2_vals, l_vals, Z_plot, levels=20, cmap="viridis")
    ax.contour(sig_f2_vals, l_vals, Z_plot, levels=20,
               colors="black", linewidths=0.4, alpha=0.5)

    # Red cross at true hyperparameters
    ax.plot(true_sigma_f2, true_l, "r+", markersize=14, markeredgewidth=2.5,
            label="True hyperparameters", zorder=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma^2$  (Variance)", fontsize=10)
    ax.set_ylabel(r"$\ell$  (Lengthscale)", fontsize=10)
    ax.set_title(f"N = {n_train}", fontsize=11)
    plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)

axes[0].legend(fontsize=8, loc="upper left")

fig.tight_layout()
plt.show()
plt.savefig("ridge.png",
            dpi=150, bbox_inches="tight")
print("Saved.")

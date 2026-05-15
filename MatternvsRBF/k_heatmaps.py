"""
Heatmaps of K and K^{-1} at three length scales:
  l_d  (focused spacing)
  l_s  (static spacing)
  l*   (MLE optimum)

Shows visually how the block structure changes and why
alpha_S is insensitive to l while alpha_D is not.
"""
import numpy as np
from numpy.linalg import solve, slogdet
from scipy.optimize import differential_evolution
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

Pe    = 500
sigma = 1.0
sn    = 1e-3

def u_true(x):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1) + np.sin(np.pi * x)

def rbf(x, y, l):
    r2 = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * r2 / l**2)

# Config A: n_s=25, n_d=25
n_s, n_d = 25,10
n        = n_s + n_d

x_s   = np.linspace(0.00, 0.90, n_s)
x_d   = np.linspace(0.92, 1.00, n_d)
x_all = np.concatenate([x_s, x_d])
y_all = np.concatenate([u_true(x_s), u_true(x_d)])

l_s_val = (x_s[-1] - x_s[0]) / (n_s - 1)   # 0.10
l_d_val = (x_d[-1] - x_d[0]) / (n_d - 1)   # 0.00333

# Find l* via DE
def nlml_obj(lr):
    l = np.exp(lr[0])
    K = rbf(x_all, x_all, l) + sn**2 * np.eye(n)
    try:
        alpha = solve(K, y_all)
        s, ld = slogdet(K)
        return 1e10 if s <= 0 else 0.5*(y_all@alpha + ld)
    except:
        return 1e10

res  = differential_evolution(nlml_obj, [(-13, -1)], seed=7,
                               maxiter=2000, tol=1e-14, popsize=20)
l_opt = np.exp(res.x[0])

print(f"l_d = {l_d_val:.5f},  l_s = {l_s_val:.5f},  l* = {l_opt:.5f}")

# ── Compute K and K^{-1} at three scales ──────────────────────────────────────
scales = {
    f'$\\ell_d$ = {l_d_val:.4f}\n(focused spacing)': l_d_val,
    f'$\\ell^*$ = {l_opt:.4f}\n(MLE optimum)':     l_opt,
    f'$\\ell_s$ = {l_s_val:.4f}\n(static spacing)':  l_s_val,
}

matrices = {}
for label, l in scales.items():
    K    = rbf(x_all, x_all, l) + sn**2 * np.eye(n)
    Kinv = np.linalg.inv(K)
    matrices[label] = (K, Kinv, l)

# ── Plotting ───────────────────────────────────────────────────────────────────
# Layout: 3 columns (one per scale), 3 rows:
#   row 0: K
#   row 1: K^{-1}  (raw, signed)
#   row 2: |K^{-1}|  (log scale to show structure across orders of magnitude)

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.patch.set_facecolor('#f5f5f5')

# Separator position between the two groups (after index n_s-1)
sep = n_s - 0.5

# Colormaps
cmap_K    = 'Blues'
cmap_Kinv = 'RdBu_r'
cmap_abs  = 'YlOrRd'

col_labels = list(scales.keys())

for col_idx, (label, (K, Kinv, l)) in enumerate(matrices.items()):

    # ── Row 0: K ──────────────────────────────────────────────────────────────
    ax = axes[0, col_idx]
    im = ax.imshow(K, cmap=cmap_K, aspect='equal',
                   vmin=0, vmax=sigma**2 + sn**2)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 format=ticker.FormatStrFormatter('%.3f'))
    ax.axhline(sep, color='red', lw=1.5, ls='--')
    ax.axvline(sep, color='red', lw=1.5, ls='--')
    ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
    if col_idx == 0:
        ax.set_ylabel('$K$', fontsize=13, fontweight='bold')
    ax.set_xticks([n_s//2, n_s + n_d//2])
    ax.set_xticklabels(['static\n(S)', 'focused\n(D)'], fontsize=8)
    ax.set_yticks([n_s//2, n_s + n_d//2])
    ax.set_yticklabels(['S', 'D'], fontsize=8)

    # Block annotations
    for (r0, r1, c0, c1, txt) in [
        (0, n_s, 0, n_s, 'K_SS'),
        (n_s, n, n_s, n, 'K_DD'),
        (0, n_s, n_s, n, 'K_SD'),
        (n_s, n, 0, n_s, 'K_DS'),
    ]:
        cx = (c0 + c1) / 2 - 0.5
        cy = (r0 + r1) / 2 - 0.5
        mean_val = np.mean(K[r0:r1, c0:c1])
        ax.text(cx, cy, f'{txt}\n{mean_val:.3f}',
                ha='center', va='center', fontsize=7,
                color='white' if mean_val > 0.5 else 'black',
                fontweight='bold')

    # ── Row 1: K^{-1} (signed, symmetric colorscale) ──────────────────────────
    ax = axes[1, col_idx]
    abs_max = min(np.percentile(np.abs(Kinv), 98), 1e4)
    im = ax.imshow(Kinv, cmap=cmap_Kinv, aspect='equal',
                   vmin=-abs_max, vmax=abs_max)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    ax.axhline(sep, color='red', lw=1.5, ls='--')
    ax.axvline(sep, color='red', lw=1.5, ls='--')
    if col_idx == 0:
        ax.set_ylabel('$K^{-1}$ (signed)', fontsize=12, fontweight='bold')
    ax.set_xticks([n_s//2, n_s + n_d//2])
    ax.set_xticklabels(['S', 'D'], fontsize=8)
    ax.set_yticks([n_s//2, n_s + n_d//2])
    ax.set_yticklabels(['S', 'D'], fontsize=8)

    # Annotate with diagonal and off-diagonal means per block
    for (r0, r1, c0, c1, tag) in [
        (0,   n_s, 0,   n_s, 'SS'),
        (n_s, n,   n_s, n,   'DD'),
    ]:
        block = Kinv[r0:r1, c0:c1].copy()
        diag_mean  = np.mean(np.diag(block))
        mask = np.ones_like(block, dtype=bool)
        np.fill_diagonal(mask, False)
        off_mean = np.mean(np.abs(block[mask]))
        cx = (c0 + c1) / 2 - 0.5
        cy = (r0 + r1) / 2 - 0.5
        ax.text(cx, cy,
                f'diag≈{diag_mean:.2f}\n|off|≈{off_mean:.2f}',
                ha='center', va='center', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # ── Row 2: log10(|K^{-1}|) to show magnitude structure ───────────────────
    ax = axes[2, col_idx]
    log_abs = np.log10(np.abs(Kinv) + 1e-15)
    im = ax.imshow(log_abs, cmap=cmap_abs, aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('log₁₀|K⁻¹|', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.axhline(sep, color='white', lw=1.5, ls='--')
    ax.axvline(sep, color='white', lw=1.5, ls='--')
    if col_idx == 0:
        ax.set_ylabel('$\\log_{10}|K^{-1}|$', fontsize=12, fontweight='bold')
    ax.set_xticks([n_s//2, n_s + n_d//2])
    ax.set_xticklabels(['S', 'D'], fontsize=8)
    ax.set_yticks([n_s//2, n_s + n_d//2])
    ax.set_yticklabels(['S', 'D'], fontsize=8)

    # Annotate with mean log-magnitude per block
    for (r0, r1, c0, c1, tag) in [
        (0,   n_s, 0,   n_s, 'K⁻¹_SS'),
        (n_s, n,   n_s, n,   'K⁻¹_DD'),
        (0,   n_s, n_s, n,   'K⁻¹_SD'),
    ]:
        val = np.mean(log_abs[r0:r1, c0:c1])
        cx  = (c0 + c1) / 2 - 0.5
        cy  = (r0 + r1) / 2 - 0.5
        ax.text(cx, cy, f'{tag}\n10^{val:.1f}',
                ha='center', va='center', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

# Row labels on far right
for row_idx, row_title in enumerate(['$K$', '$K^{-1}$ (signed)',
                                      '$\\log_{10}|K^{-1}|$']):
    axes[row_idx, -1].yaxis.set_label_position('right')

fig.suptitle(
    f'Block structure of $K$ and $K^{{-1}}$ at three length scales\n'
    f'n_s={n_s} static (S, left blocks) | n_d={n_d} focused (D, right blocks) | '
    f'red dashes = group boundary',
    fontsize=12, fontweight='bold', y=1.01
)

plt.tight_layout()


# ── Second figure: alpha vector heatmap ───────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig2.patch.set_facecolor('#f5f5f5')

for col_idx, (label, (K, Kinv, l)) in enumerate(matrices.items()):
    alpha = Kinv @ y_all
    ax    = axes2[col_idx]

    # Plot as a column heatmap: n x 1
    vals  = alpha.reshape(-1, 1)
    abs_max = min(np.percentile(np.abs(alpha), 99), 1e4)
    im = ax.imshow(vals, cmap='RdBu_r', aspect='auto',
                   vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, fraction=0.15, pad=0.04)

    ax.axhline(sep, color='black', lw=2, ls='--')
    ax.set_title(label + f'\nmean|α_S|={np.mean(np.abs(alpha[:n_s])):.4f}'
                        f'\nmean|α_D|={np.mean(np.abs(alpha[n_s:])):.2f}',
                 fontsize=9, fontweight='bold')
    ax.set_yticks([n_s//2, n_s + n_d//2])
    ax.set_yticklabels(['Static\n(S)', 'Focused\n(D)'], fontsize=9)
    ax.set_xticks([])

    # Annotate each entry
    for i, v in enumerate(alpha):
        color = 'white' if abs(v) > abs_max * 0.5 else 'black'
        ax.text(0, i, f'{v:.2f}', ha='center', va='center',
                fontsize=5.5, color=color)


plt.tight_layout()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
import warnings
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = "browser"
warnings.filterwarnings('ignore')



V_TRUE     = 5.0
KAPPA_TRUE = 0.01
Pe         = V_TRUE / KAPPA_TRUE
DELTA      = KAPPA_TRUE / V_TRUE      # boundary layer thickness ~ 0.002

N_OBS      = 80    # scattered interior observations
N_STATIC_C = 120   # interior collocation points (uniform grid)
N_BC       = 60    # boundary condition points (15 per edge)
N_STAR_1D  = 40    # prediction grid per axis => N_STAR_1D^2 total

NOISE_OBS  = 1e-3
NOISE_COL  = 1e-3

np.random.seed(42)

# ─────────────────────────────────────────────
#  EXACT SOLUTION
# ─────────────────────────────────────────────

def u1d_exact(x, kappa, v):
    """Exact solution of  v*U' - kappa*U'' + kappa*pi^2*U = sin(pi*x),  U(0)=U(1)=0."""
    pi = np.pi
    denom = pi**2 * kappa**2 + v**2
    A  =  kappa / denom
    B  = -v     / (pi * denom)
    r  = v / kappa
    e  = np.exp(r)
    C2 = 2*B / (e - 1)
    C1 = -B - C2
    return C1 + C2*np.exp(r*x) + A*np.sin(pi*x) + B*np.cos(pi*x)

def u_exact_2d(xy, kappa, v):
    x, y = xy[:, 0], xy[:, 1]
    u2d = u1d_exact(x, kappa, v) * np.sin(np.pi * y)
    return u2d

def f_rhs_2d(xy):
    """Prescribed forcing  f(x,y) = sin(pi*x)*sin(pi*y)."""
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])



def _rbf_factors(p, q, lx, ly, sigma):
    """Shared factors for all kernel blocks. Returns (K, rx, ry, ex, ey, lx2, ly2)."""
    rx  = p[:, 0][:, None] - q[:, 0][None, :]   # (M, N)
    ry  = p[:, 1][:, None] - q[:, 1][None, :]
    lx2, ly2 = lx**2, ly**2
    K   = sigma**2 * np.exp(-0.5*rx**2/lx2) * np.exp(-0.5*ry**2/ly2)
    ex  = rx / lx2
    ey  = ry / ly2
    return K, rx, ry, ex, ey, lx2, ly2


def k2d(p, q, lx, ly, sigma):
    """Plain product-RBF kernel, shape (M, N)."""
    K, *_ = _rbf_factors(p, q, lx, ly, sigma)
    return K


def k2d_L2(p, q, lx, ly, sigma, v, kappa):

    K, rx, ry, ex, ey, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)
    Lq = v*ex - kappa*(ex**2 - 1.0/lx2) - kappa*(ey**2 - 1.0/ly2)
    return Lq * K


def k2d_L1L2(p, q, lx, ly, sigma, v, kappa):

    K, rx, ry, ex, ey, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)

    Lq        = v*ex - kappa*(ex**2 - 1.0/lx2) - kappa*(ey**2 - 1.0/ly2)
    dLq_dx1   = (v - 2*kappa*ex) / lx2
    d2Lq_dx12 = -2*kappa / lx2**2
    dLq_dy1   = -2*kappa*ey / ly2
    d2Lq_dy12 = -2*kappa / ly2**2

    dF_dx1    = (dLq_dx1 - ex*Lq)
    d2F_dx12  = (d2Lq_dx12 - Lq/lx2 - 2*ex*dLq_dx1 + ex**2*Lq)
    d2F_dy12  = (d2Lq_dy12 - Lq/ly2 - 2*ey*dLq_dy1 + ey**2*Lq)

    return (v*dF_dx1 - kappa*(d2F_dx12 + d2F_dy12)) * K



def build_gpr_system(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                     lx, ly, sigma, sigma_n, v, kappa):
    """
    Assemble training targets and kernel matrix.

    Row/column layout:
      [ BC points (value) | observations (value) | collocation (operator) ]

    Returns: xy_val, y_train, K_train, n_v
      xy_val  : (n_v, 2) value-type input points (BC + obs)
      y_train : (n,)     all targets
      K_train : (n, n)   kernel matrix with noise
      n_v     : number of value-type rows
    """
    xy_val  = np.vstack([xy_bc, xy_obs])
    u_val   = np.concatenate([u_bc, u_obs])
    n_v, n_c = len(xy_val), len(xy_col)
    n        = n_v + n_c
    y_train  = np.concatenate([u_val, f_col])

    K_vv = k2d(xy_val, xy_val, lx, ly, sigma)
    K_vc = k2d_L2(xy_val, xy_col, lx, ly, sigma, v, kappa)
    K_cc = k2d_L1L2(xy_col, xy_col, lx, ly, sigma, v, kappa)

    K = np.zeros((n, n))
    K[:n_v, :n_v] = K_vv
    K[:n_v, n_v:] = K_vc
    K[n_v:, :n_v] = K_vc.T
    K[n_v:, n_v:] = K_cc
    K += sigma_n**2 * np.eye(n)
    K += 1e-8      * np.eye(n)   # jitter

    return xy_val, y_train, K, n_v


def gpr_posterior(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                  xy_star, lx, ly, sigma, sigma_n, v, kappa):
    xy_val, y_train, K_train, n_v = build_gpr_system(
        xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
        lx, ly, sigma, sigma_n, v, kappa)

    L_fac  = cho_factor(K_train)
    alpha  = cho_solve(L_fac, y_train)

    # Cross-covariance k(xy_star, training):
    #   value columns  -> plain kernel
    #   operator cols  -> L on second argument
    Ks = np.hstack([
        k2d(xy_star, xy_val,  lx, ly, sigma),
        k2d_L2(xy_star, xy_col, lx, ly, sigma, v, kappa),
    ])                                         # (n_star, n_train)

    mu  = Ks @ alpha
    V   = cho_solve(L_fac, Ks.T)
    var = sigma**2 * np.ones(len(xy_star)) - np.einsum('ij,ji->i', Ks, V)
    var = np.maximum(var, 0.0)

    return mu, var


def gpr_nlml(theta, xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc, v, kappa):
    """Negative log marginal likelihood (with log-normal priors)."""
    lx, ly, sigma, log_sn = theta
    sn = np.exp(log_sn)
    _, y_train, K_train, _ = build_gpr_system(
        xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
        lx, ly, sigma, sn, v, kappa)
    sign, ld = slogdet(K_train)
    if sign <= 0:
        return 1e10
    alpha = solve(K_train, y_train)
    nlml  = 0.5 * (y_train @ alpha) + 0.5 * ld
    # Log-normal priors on lx, ly, sigma, sigma_n
    nlml -= -0.5 * ((np.log(lx)    - np.log(0.15)) / 0.8)**2
    nlml -= -0.5 * ((np.log(ly)    - np.log(0.15)) / 0.8)**2
    nlml -= -0.5 * ((np.log(sigma) - np.log(1.0))  / 0.5)**2
    nlml -= -0.5 * ((log_sn        - np.log(1e-3)) / 1.0)**2
    return nlml



xy_obs = np.column_stack([
    np.random.uniform(0.0, 1.0 , N_OBS),
    np.random.uniform(0.0, 1.0,            N_OBS),
])
u_obs = u_exact_2d(xy_obs, KAPPA_TRUE, V_TRUE) \
      + np.random.normal(0, NOISE_OBS, N_OBS)

# Static interior collocation points (uniform grid, interior only)
nc_1d = int(np.ceil(np.sqrt(N_STATIC_C)))
_cx = np.linspace(0, 1, nc_1d + 2)[1:-1]
_cy = np.linspace(0, 1, nc_1d + 2)[1:-1]
_CX, _CY = np.meshgrid(_cx, _cy)
xy_col = np.column_stack([_CX.ravel(), _CY.ravel()])
f_col  = f_rhs_2d(xy_col) + np.random.normal(0, NOISE_COL, len(xy_col))

# Boundary condition points: 15 per edge, u=0
n_per_edge = N_BC // 4
_e    = np.linspace(0, 1, n_per_edge)
xy_bc = np.vstack([
    np.column_stack([_e,                   np.zeros(n_per_edge)]),  # y=0
    np.column_stack([_e,                   np.ones(n_per_edge)]),   # y=1
    np.column_stack([np.zeros(n_per_edge), _e]),                    # x=0
    np.column_stack([np.ones(n_per_edge),  _e]),                    # x=1
])
u_bc = np.zeros(len(xy_bc))

# Prediction grid
_gx = np.linspace(0, 1, N_STAR_1D)
_gy = np.linspace(0, 1, N_STAR_1D)
_GX, _GY = np.meshgrid(_gx, _gy)
xy_star = np.column_stack([_GX.ravel(), _GY.ravel()])
u_true  = u_exact_2d(xy_star, KAPPA_TRUE, V_TRUE)

print("Problem setup (2D):")
print(f"  Pe={Pe:.0f},  delta={DELTA:.4f}")
print(f"  n_obs={N_OBS}, n_col={len(xy_col)}, n_bc={len(xy_bc)}")
print(f"  Prediction grid: {N_STAR_1D}x{N_STAR_1D} = {len(xy_star)} points")


THETA0 = np.array([0.15, 0.15, 1.0, np.log(1e-3)])   # lx, ly, sigma, log_sn
BOUNDS = [(0.02, 0.5), (0.02, 0.5), (0.1, 5.0), (-10, -1)]


print(f"\n{'='*55}")
print("  Optimising hyperparameters...")
res = minimize(
    gpr_nlml, THETA0,
    args=(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc, V_TRUE, KAPPA_TRUE),
    method='L-BFGS-B', bounds=BOUNDS,
    options={'maxiter': 300, 'ftol': 1e-12},
)
lx_opt, ly_opt, sigma_opt, log_sn_opt = res.x
sn_opt = np.exp(log_sn_opt)
print(f"  lx={lx_opt:.4f}, ly={ly_opt:.4f}, sigma={sigma_opt:.4f}, sn={sn_opt:.2e}")

print("  Computing posterior...")
mu, var = gpr_posterior(
    xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
    xy_star, lx_opt, ly_opt, sigma_opt, sn_opt, V_TRUE, KAPPA_TRUE)
std  = np.sqrt(var)
err  = np.abs(mu - u_true)
l2   = np.linalg.norm(mu - u_true) / np.linalg.norm(u_true)
print(f"  L2 relative error: {l2:.4f}")


SZ = N_STAR_1D

def to_grid(a): return a.reshape(SZ, SZ)

GX = _GX; GY = _GY
u_true_g = to_grid(u_true)
mu_g     = to_grid(mu)
std_g    = to_grid(std)
err_g    = to_grid(err)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.patch.set_facecolor('#f8f9fa')

BL_LEFT  = max(0.0, 1.0 - 15*DELTA)   
VRANGE   = np.nanmax(np.abs(u_true_g))

def _pcolor(ax, data, title, cmap='RdYlBu_r', vmin=None, vmax=None, cb_label=''):
    ax.set_facecolor('white')
    if vmin is None and vmax is None:
        vm   = np.nanmax(np.abs(data))
        vmin, vmax = -vm, vm
    pcm = ax.pcolormesh(GX, GY, data, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto', rasterized=True)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=cb_label)
    ax.axvspan(BL_LEFT, 1.0, alpha=0.08, color='orange')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Row 0 ── solution fields
_pcolor(axes[0, 0], u_true_g, 'Exact  u(x,y)',
        cmap='RdYlBu_r', vmin=-VRANGE, vmax=VRANGE, cb_label='u')
axes[0, 0].scatter(xy_obs[:, 0], xy_obs[:, 1],
                   s=10, c='lime', edgecolors='k', lw=0.3, zorder=5, label='obs')
axes[0, 0].legend(fontsize=7, loc='upper left')

_pcolor(axes[0, 1], mu_g, f'GPR posterior mean  (L2={l2:.3f})',
        cmap='RdYlBu_r', vmin=-VRANGE, vmax=VRANGE, cb_label='u')

_pcolor(axes[0, 2], err_g, 'Pointwise |error|',
        cmap='hot_r', vmin=0, vmax=np.nanmax(err_g), cb_label='|u−û|')

# Row 1 ── uncertainty + point layout + full slice
_pcolor(axes[1, 0], std_g, 'Posterior std  σ(x,y)',
        cmap='Oranges', vmin=0, vmax=np.nanmax(std_g), cb_label='σ')

ax_pts = axes[1, 1]
ax_pts.set_facecolor('white')
ax_pts.scatter(xy_obs[:, 0],  xy_obs[:, 1],  s=10, c='lime',      edgecolors='k', lw=0.3,
               zorder=4, label=f'Obs ({N_OBS})')
ax_pts.scatter(xy_col[:, 0],  xy_col[:, 1],  s=5,  c='purple',    edgecolors='none',
               zorder=3, alpha=0.6, marker='s', label=f'Colloc ({len(xy_col)})')
ax_pts.scatter(xy_bc[:, 0],   xy_bc[:, 1],   s=10, c='royalblue', edgecolors='k', lw=0.3,
               zorder=5, label=f'BC ({len(xy_bc)})')
ax_pts.axvspan(BL_LEFT, 1.0, alpha=0.08, color='orange')
ax_pts.set_title('Training point layout', fontsize=10, fontweight='bold')
ax_pts.set_xlabel('x'); ax_pts.set_ylabel('y')
ax_pts.set_xlim(0, 1); ax_pts.set_ylim(0, 1)
ax_pts.legend(fontsize=7, loc='upper left')
ax_pts.grid(True, alpha=0.3)

y_mid_idx = SZ // 2
x_plot    = _gx

ax_sl = axes[1, 2]
ax_sl.set_facecolor('white')
ax_sl.plot(x_plot, u_true_g[y_mid_idx], '--', color='#1a1a2e', lw=2.5, label='Exact')
ax_sl.plot(x_plot, mu_g[y_mid_idx],      '-', color='#e94560', lw=2,   label='GPR mean')
ax_sl.fill_between(x_plot,
    mu_g[y_mid_idx] - 2*std_g[y_mid_idx],
    mu_g[y_mid_idx] + 2*std_g[y_mid_idx],
    alpha=0.20, color='#e94560', label='±2σ')
ax_sl.axvspan(BL_LEFT, 1.0, alpha=0.08, color='orange', label='BL region')
ax_sl.set_title('Mid-plane slice  y=0.5  (full domain)', fontsize=10, fontweight='bold')
ax_sl.set_xlabel('x'); ax_sl.set_ylabel('u(x, 0.5)')
ax_sl.legend(fontsize=8); ax_sl.grid(True, alpha=0.3)

# Row 2 ── boundary layer zoom + error slice + spare
ax_zoom = axes[2, 0]
ax_zoom.set_facecolor('white')
zm = x_plot >= BL_LEFT - 0.05
ax_zoom.plot(x_plot[zm], u_true_g[y_mid_idx][zm], '--', color='#1a1a2e', lw=2.5, label='Exact')
ax_zoom.plot(x_plot[zm], mu_g[y_mid_idx][zm],      '-', color='#e94560', lw=2,   label='GPR mean')
ax_zoom.fill_between(x_plot[zm],
    (mu_g[y_mid_idx] - 2*std_g[y_mid_idx])[zm],
    (mu_g[y_mid_idx] + 2*std_g[y_mid_idx])[zm],
    alpha=0.25, color='#e94560', label='±2σ')
ax_zoom.axvspan(BL_LEFT, 1.0, alpha=0.12, color='orange')
ax_zoom.set_title('Mid-plane slice  y=0.5  (boundary layer zoom)', fontsize=10, fontweight='bold')
ax_zoom.set_xlabel('x'); ax_zoom.set_ylabel('u(x, 0.5)')
ax_zoom.set_xlim(BL_LEFT - 0.05, 1.0)
ax_zoom.legend(fontsize=8); ax_zoom.grid(True, alpha=0.3)

ax_esl = axes[2, 1]
ax_esl.set_facecolor('white')
ax_esl.semilogy(x_plot, err_g[y_mid_idx], color='#e94560', lw=2)
ax_esl.axvspan(BL_LEFT, 1.0, alpha=0.08, color='orange', label='BL region')
ax_esl.set_title('Mid-plane |error|  y=0.5', fontsize=10, fontweight='bold')
ax_esl.set_xlabel('x'); ax_esl.set_ylabel('|u−û|')
ax_esl.legend(fontsize=8); ax_esl.grid(True, alpha=0.3, which='both')

# hide the unused subplot
axes[2, 2].set_visible(False)

plt.suptitle(
    f'2D Advection-Diffusion  [0,1]²  |  Pe={Pe:.0f}  |  '
    f'v=({V_TRUE},0), κ={KAPPA_TRUE},  δ≈{DELTA:.4f}\n'
    f'RBF GPR  |  lx={lx_opt:.3f}, ly={ly_opt:.3f}, '
    f'σ={sigma_opt:.3f}, σₙ={sn_opt:.1e}  |  L2={l2:.4f}',
    fontsize=12, fontweight='bold',
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.40, wspace=0.35)
plt.savefig('advdiff_2d_gpr.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: advdiff_2d_gpr.png")
plt.show()
print("Done.")

fig3d = go.Figure(
    data=[go.Surface(
        x=_GX,
        y=_GY,
        z=mu_g,
        colorscale='Viridis',
        colorbar=dict(title='u'))]
)
fig3d.update_layout(
    title='Combined posterior mean μ₂(x,y) — interactive 3D surface',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='u(x,y)'),
    autosize=True,
    margin=dict(l=0, r=0, t=40, b=0)
)
fig3d.show()
fig3d.write_html('advdiff_2d_combined_3d.html')

fig3d = go.Figure(
    data=[go.Surface(
        x=_GX,
        y=_GY,
        z=u_true_g,
        colorscale='Viridis',
        colorbar=dict(title='u'))]
)
fig3d.update_layout(
    title='True PDE solution — interactive 3D surface',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='u(x,y)'),
    autosize=True,
    margin=dict(l=0, r=0, t=40, b=0)
)
fig3d.show()
fig3d.write_html('advdiff_2d_combined_3d.html')
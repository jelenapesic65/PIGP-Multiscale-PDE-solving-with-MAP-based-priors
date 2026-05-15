import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import slogdet, solve, cholesky
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
import warnings
warnings.filterwarnings('ignore')


# PROBLEM SETUP  (Interior-oscillation regime)
#
# Forcing:  f(x) = sin(3*pi*x) + 2*sin(7*pi*x) - 1.5*cos(5*pi*x) + sin(15*pi*x)
# This drives 7+ interior extrema — the solution changes strongly throughout
# the bulk, NOT only at the boundary layer.
#
# Parameters: v=0.5, kappa=0.01  =>  Pe=50, delta=0.02
# Pe is moderate so the interior features dominate, yet a thin BL still exists.
# No closed-form solution: solved numerically via sparse central-difference FD.

V_TRUE     = 5          # advection velocity
KAPPA_TRUE = 0.01         # diffusion coefficient
Pe         = V_TRUE / KAPPA_TRUE         # = 50
DELTA      = KAPPA_TRUE / V_TRUE         # = 0.02  (BL thickness)

DOMAIN     = (0.0, 1.0)
N_OBS      = 30          # interior observations
N_STATIC_C = 80           # static collocation points (more to resolve oscillations)
N_DYN      = 40           # dynamic collocation points (placed at high-residual zones)
N_STAR     = 300          # prediction points
N_FD       = 8000         # FD grid points for reference solution
NOISE_OBS  = 1e-3
NOISE_COL  = 1e-3
np.random.seed(42)

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline

def f_rhs(x):
    """Multi-frequency forcing — drives strong oscillations in the interior."""
    return (np.sin(3*np.pi*x)
            + 2.0*np.sin(7*np.pi*x)
            - 1.5*np.cos(5*np.pi*x)
            + np.sin(15*np.pi*x))

def solve_pde_fd(v, kappa, f_func, n=N_FD):
    """
    Solve  v*u' - kappa*u'' = f(x)  on [0,1],  u(0)=u(1)=0
    via second-order central-difference sparse linear system.
    Cell Pe = v*h/kappa << 1 ensures stability.
    """
    x_all = np.linspace(0.0, 1.0, n + 2)
    h     = x_all[1] - x_all[0]
    xi    = x_all[1:-1]                         # n interior nodes
    a =  -v/(2*h) - kappa/h**2                  # sub-diagonal
    d =   2*kappa/h**2                          # diagonal
    c =   v/(2*h) - kappa/h**2                  # super-diagonal
    A = diags([a*np.ones(n-1), d*np.ones(n), c*np.ones(n-1)],
              [-1, 0, 1], format="csr")
    u = spsolve(A, f_func(xi))
    return xi, u

# Build reference solution and its cubic-spline interpolant
_x_fd, _u_fd = solve_pde_fd(V_TRUE, KAPPA_TRUE, f_rhs)
_cs_ref       = CubicSpline(_x_fd, _u_fd, bc_type="not-a-knot")

def u_exact(x, kappa=None, mu=None):
    """Numerical reference solution via cubic-spline interpolation of FD solution."""
    return _cs_ref(np.asarray(x))

# KERNEL LIBRARY  (RBF only)

def rbf_k(x, y, l, sigma):
    r2 = (x[:, None] - y[None, :])**2
    return sigma**2 * np.exp(-0.5 * r2 / l**2)

def rbf_derivs(x, y, l, sigma):
  
    r  = x[:, None] - y[None, :]
    k  = sigma**2 * np.exp(-0.5 * r**2 / l**2)
    l2, l4, l6, l8 = l**2, l**4, l**6, l**8
    k_x    = -(r/l2) * k
    k_y    =  (r/l2) * k
    k_xx   = (r**2/l4 - 1/l2) * k
    k_yy   = k_xx
    k_xy   = (1/l2 - r**2/l4) * k
    k_xxy  = (-3*r/l4 + r**3/l6) * k
    k_xyy  =  (3*r/l4 - r**3/l6) * k
    k_xxyy = (3/l4 - 6*r**2/l6 + r**4/l8) * k
    return k, k_x, k_y, k_xx, k_yy, k_xy, k_xxy, k_xyy, k_xxyy

def get_derivs(kernel_name, x, y, l, sigma):
    return rbf_derivs(x, y, l, sigma)

def get_kernel(kernel_name, x, y, l, sigma):
    return rbf_k(x, y, l, sigma)



# STANDARD GPR PIPELINE  (Pipeline 1)

def build_gpr_system(x_obs, y_obs, x_col, y_col,
                     l, sigma, sigma_n, v, kappa,
                     kernel_name='rbf'):
    """
    Assemble GPR training data with PDE operator-informed collocation.

    Training rows:
      [BC_left, BC_right, interior observations, collocation points]

    BC and observation rows evaluate the GP value directly.
    Collocation rows apply the PDE operator L = v*d/dx - kappa*d^2/dx^2
    to the GP, so their covariance is computed via kernel derivatives.

    Returns: x_train, y_train, K_train, x_val, x_pde, n_v
    """
    x_bc = np.array([0.0, 1.0])
    y_bc = np.array([0.0, 0.0])

    x_val   = np.concatenate([x_bc, x_obs])   # value-type training points
    y_val   = np.concatenate([y_bc, y_obs])
    x_pde   = x_col                            # operator-type training points
    y_pde   = y_col

    n_v = len(x_val)
    n_c = len(x_pde)
    n   = n_v + n_c

    y_train = np.concatenate([y_val, y_pde])

    k, *_ = get_derivs(kernel_name, x_val, x_val, l, sigma)
    K_vv = k

    # Block (val, pde)  L applied to second argument
    #   K_vc[i,j] = Cov(u(x_val_i), L[u](x_pde_j))
    #             = v * k_y(x_val, x_pde) - kappa * k_yy(x_val, x_pde)
    k, k_x, k_y, k_xx, k_yy, *_ = get_derivs(kernel_name, x_val, x_pde, l, sigma)
    K_vc = v * k_y - kappa * k_yy

    # Block (pde, pde) : L^T L applied to both arguments ---
    #   K_cc = v^2 k_xy - v*kappa*(k_xyy + k_xxy) + kappa^2 k_xxyy
    k, k_x, k_y, k_xx, k_yy, k_xy, k_xxy, k_xyy, k_xxyy = \
        get_derivs(kernel_name, x_pde, x_pde, l, sigma)
    K_cc = (v**2   * k_xy
            - v * kappa * (k_xyy + k_xxy)
            + kappa**2  * k_xxyy)

    K_train = np.zeros((n, n))
    K_train[:n_v, :n_v] = K_vv
    K_train[:n_v, n_v:] = K_vc
    K_train[n_v:, :n_v] = K_vc.T
    K_train[n_v:, n_v:] = K_cc

    K_train += sigma_n**2 * np.eye(n)
    K_train += 1e-8 * np.eye(n)

    return x_val, x_pde, y_train, K_train, n_v


def gpr_posterior(x_obs, y_obs, x_col, y_col,
                  x_star, l, sigma, sigma_n, v, kappa,
                  kernel_name='rbf'):
    """Standard GPR posterior mean and variance at x_star."""
    x_val, x_pde, y_train, K_train, n_v = \
        build_gpr_system(x_obs, y_obs, x_col, y_col,
                         l, sigma, sigma_n, v, kappa, kernel_name)

    L_fac = cho_factor(K_train)
    alpha = cho_solve(L_fac, y_train)

    # Cross-covariance k(x_star, x_train):
    #   value columns  -> k(x_star, x_val)
    #   operator cols  -> L_y k(x_star, x_pde)
    k, k_x, k_y, k_xx, k_yy, *_ = get_derivs(kernel_name, x_star, x_val, l, sigma)
    Ks_val = k

    k, k_x, k_y, k_xx, k_yy, *_ = get_derivs(kernel_name, x_star, x_pde, l, sigma)
    Ks_pde = v * k_y - kappa * k_yy

    K_star = np.hstack([Ks_val, Ks_pde])   # (n_star, n_train)

    mu_star = K_star @ alpha

    K_ss_diag = sigma**2 * np.ones(len(x_star))
    V = cho_solve(L_fac, K_star.T)
    var_star = K_ss_diag - np.einsum('ij,ji->i', K_star, V)
    var_star = np.maximum(var_star, 0.0)

    return mu_star, var_star, K_train, alpha


def gpr_nlml(theta, x_obs, y_obs, x_col, y_col, kernel_name):
    l, sigma, log_sn, v, kap = theta
    sn = np.exp(log_sn)
    x_val, x_pde, y_train, K_train, n_v = \
        build_gpr_system(x_obs, y_obs, x_col, y_col,
                         l, sigma, sn, v, kap, kernel_name)
    sign, ld = slogdet(K_train)
    if sign <= 0:
        return 1e10
    alpha = solve(K_train, y_train)
    nlml  = 0.5 * y_train @ alpha + 0.5 * ld
    nlml -= -0.5 * ((np.log(l)    - np.log(0.1)) / 0.8)**2
    nlml -= -0.5 * ((np.log(sigma) - np.log(1.0)) / 0.5)**2
    nlml -= -0.5 * ((log_sn       - np.log(1e-3)) / 1.0)**2
    return nlml

# GPR CORRECTION PIPELINE


def gpr_correction(x_dyn, y_dyn, mu_prior_dyn, var_prior_dyn,
                   x_star, mu_prior_star, var_prior_star,
                   l_loc, sigma_loc, sigma_n_loc,
                   kernel_name='rbf',
                   dyn_left=0.0, dyn_right=1.0):
    """
    GPR on residual:  r(x) = u(x) - mu_prior(x)
    Observations:     r_obs = y_dyn - mu_prior_dyn

    The correction is localised: outside the dynamic region the correction
    mean decays naturally through the kernel, but we additionally taper it
    using a smooth window so that far-field oscillations (especially bad for
    RBF) do not degrade the already-good GPR solution.

    Returns corrected mean and variance at x_star.
    """
    r_obs = y_dyn - mu_prior_dyn


    # Build kernel matrices for residual GP
    Kdd = get_kernel(kernel_name, x_dyn, x_dyn, l_loc, sigma_loc)
    # FIX BUG 4: must go on diagonal only — scalar addition corrupts all off-diagonals
    Kdd += (sigma_n_loc**2 + np.mean(var_prior_dyn) + 1e-8) * np.eye(len(x_dyn))

    Ksd = get_kernel(kernel_name, x_star, x_dyn, l_loc, sigma_loc)
    Kss_diag = sigma_loc**2 * np.ones(len(x_star))

    L     = cho_factor(Kdd)
    alpha = cho_solve(L, r_obs)

    delta_mu  = Ksd @ alpha
    v_        = cho_solve(L, Ksd.T)
    delta_var = Kss_diag - np.einsum('ij,ji->i', Ksd, v_)
    delta_var = np.maximum(delta_var, 0.0)

    # ── Smooth taper: correction fades to zero outside the dynamic region.
    #
    # FIX — three bugs were causing the correction to blow up:
    #
    # BUG 1 (taper width):  old code set width = max(l_loc, 5*BL_width).
    #   With BL_width=0.016 and l_loc=0.03 this gave width=0.08, which made
    #   the logistic transition span ~80% of [0,1] — effectively no taper at all.
    #   The taper was ~0.49 at x=0.9, letting the wild local-GP extrapolation
    #   propagate deep into the interior where GPR was already accurate.
    #   FIX: margin = 0.5 * BL_width  (stays proportional to the zone, not l_loc)
    #
    # BUG 2 (taper slope):  old slope = width/4 was too shallow once width was
    #   miscalculated. Even with the correct width the old slope let the taper
    #   linger far outside the zone.
    #   FIX: slope = margin / 5  (much steeper, ~zero within one BL-width outside)
    #
    # BUG 3 (l_loc vs BL spacing): l_loc = max(2*delta, 0.03) = 0.03.
    #   The 40 BL points are packed in a window of width 0.016, so
    #   l_loc/spacing ≈ 78 — every point looks identical to the GP.
    #   The GP then must extrapolate a large-amplitude correction (fitting
    #   residuals up to 0.05 with sigma_loc=0.1) over l_loc=0.03 >> BL_width,
    #   producing violent oscillations outside x∈BL.
    #   FIX: l_loc is passed from the caller (set to BL_width/4 below).
    #   sigma_loc raised to 0.5 so the prior can express the needed amplitude.
    bl_width     = dyn_right - dyn_left
    taper_margin = 0.5 * bl_width          # transition band either side of zone
    taper_slope  = taper_margin / 50       # very steep cutoff: near-step function
    taper = 1.0 / (1.0 + np.exp(-(x_star - dyn_left  + taper_margin) / taper_slope)) \
          * 1.0 / (1.0 + np.exp( (x_star - dyn_right - taper_margin) / taper_slope))
    # taper ≈ 1 inside dynamic region, ≈ 0 well outside it
    delta_mu_tapered = delta_mu * taper

    mu_combined  = mu_prior_star + delta_mu_tapered
    var_combined = var_prior_star + delta_var * taper

    return mu_combined, var_combined, delta_mu_tapered, delta_var



a, b = DOMAIN

x_obs_raw = np.sort(np.random.uniform(a, b, N_OBS))
y_obs      = u_exact(x_obs_raw) + np.random.normal(0, NOISE_OBS, N_OBS)

x_col_static = np.linspace(a, b, N_STATIC_C+2)[1:-1]
y_col_static  = f_rhs(x_col_static) + np.random.normal(0, NOISE_COL, N_STATIC_C)

# Prediction grid
x_star = np.linspace(a, b, N_STAR)
y_true = u_exact(x_star)

# ── Adaptive dynamic region ──
# Run a quick coarse GPR first pass to find high-residual interior zones,
# then place the N_DYN dynamic collocation points there.
def find_high_residual_region(x, y, x_col, y_col, x_eval, l=0.08, sigma=1.0, sn=1e-2):
    """Quick GPR pass (fixed hyperparams) to locate high-PDE-residual zones."""
    mu_q, _, _, _ = gpr_posterior(x, y, x_col, y_col, x_eval,
                                   l, sigma, sn, V_TRUE, KAPPA_TRUE, "rbf")
    dx = x_eval[1] - x_eval[0]
    du  = np.gradient(mu_q, dx)
    d2u = np.gradient(du, dx)
    res = np.abs(V_TRUE*du - KAPPA_TRUE*d2u - f_rhs(x_eval))
    return res, mu_q

print("  Quick pass to locate high-residual zones...")
res_quick, _ = find_high_residual_region(
    x_obs_raw, y_obs, x_col_static, y_col_static, x_star)
DYN_LEFT   = max(0.0, 1.0 - 8 * DELTA)
DYN_RIGHT  = 1.0

# Sample dynamic points proportional to |R(x)|^0.7 (soft concentration)
#res_prob = res_quick**0.7
#res_prob /= res_prob.sum()
#dyn_idx = np.random.choice(len(x_star), size=N_DYN, replace=False, p=res_prob)
#x_dyn   = np.sort(x_star[dyn_idx])
x_dyn = np.sort(np.random.uniform(DYN_LEFT, DYN_RIGHT, N_DYN))
y_dyn   = u_exact(x_dyn) + np.random.normal(0, NOISE_OBS, N_DYN)

# Dynamic region for taper: span of the dynamic points with some margin
#DYN_LEFT  = max(a + 0.01, x_dyn.min() - 0.05)
#DYN_RIGHT = min(b - 0.01, x_dyn.max() + 0.05)

print(f"  Dynamic region set to [{DYN_LEFT:.3f}, {DYN_RIGHT:.3f}]")

print(f"Problem setup:")
print(f"  Pe = {Pe:.0f},  delta = {DELTA:.4f}")
print(f"  Boundary layer region: [{DYN_LEFT:.4f}, {DYN_RIGHT:.3f}]")
print(f"  n_obs={N_OBS}, n_static_col={N_STATIC_C}, n_dyn={N_DYN}")


THETA0_GPR = np.array([
    0.06,          # l  — shorter to capture interior oscillations (15*pi feature ~ 0.067)
    1.0,           # sigma
    np.log(1e-3),  # log sigma_n
    V_TRUE,        # v
    KAPPA_TRUE     # kappa
])

BOUNDS_GPR = [
    (0.02, 0.2),        # l — tighter upper bound; interior features need short l
    (0.1,  5.0),        # sigma
    (-10,  -1),         # log sigma_n
    (V_TRUE, V_TRUE),   # v fixed
    (KAPPA_TRUE, KAPPA_TRUE)  # kappa fixed
]

# FIX: l_loc must be << BL_width so the local GP only "sees" nearby BL points
# and does not extrapolate across the full domain.
# BL_width = DYN_RIGHT - DYN_LEFT = 8*DELTA = 0.016
# l_loc = BL_width/4 ≈ 0.004 keeps the kernel local within the BL.
L_LOCAL     = (DYN_RIGHT - DYN_LEFT) / 4   # ≈ 0.004 for this BL
SIGMA_LOCAL = 0.5    # large enough to express the ~0.05 residual amplitude
SN_LOCAL    = NOISE_OBS


results = {}

for kname in ['rbf']:#]:
    print(f"\n{'='*60}")
    print(f"  Kernel: {kname.upper()}")
    print('='*60)

    print("  Optimizing GPR hyperparameters...")
    res = minimize(
        gpr_nlml,
        THETA0_GPR,
        args=(x_obs_raw, y_obs, x_col_static, y_col_static, kname),
        method='L-BFGS-B',
        bounds=BOUNDS_GPR,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    l_opt, sigma_opt, log_sn_opt, v_opt, kap_opt = res.x
    sn_opt = np.exp(log_sn_opt)
    print(f"  l={l_opt:.4f}, sigma={sigma_opt:.4f}, sn={sn_opt:.2e}, "
          f"v={v_opt:.3f}, kappa={kap_opt:.4f}")

    print("  Running GPR pipeline...")
    (mu1, var1, K_train, alpha_gpr) = gpr_posterior(
        x_obs_raw, y_obs, x_col_static, y_col_static,
        x_star, l_opt, sigma_opt, sn_opt, v_opt, kap_opt, kname)
    std1 = np.sqrt(var1)
    print(f"  GPR L2 error: {np.linalg.norm(mu1 - y_true)/np.linalg.norm(y_true):.4f}")

    (mu1_dyn, var1_dyn, _, _) = gpr_posterior(
        x_obs_raw, y_obs, x_col_static, y_col_static,
        x_dyn, l_opt, sigma_opt, sn_opt, v_opt, kap_opt, kname)

    print("  Running GPR correction pipeline...")
    (mu2, var2, delta_mu, delta_var) = gpr_correction(
        x_dyn, y_dyn, mu1_dyn, var1_dyn,
        x_star, mu1, var1,
        L_LOCAL, SIGMA_LOCAL, SN_LOCAL, kname,
        dyn_left=DYN_LEFT, dyn_right=DYN_RIGHT)
    std2 = np.sqrt(np.maximum(var2, 0))
    print(f"  Combined L2 error: {np.linalg.norm(mu2 - y_true)/np.linalg.norm(y_true):.4f}")

    results[kname] = {
        'mu1': mu1, 'std1': std1,
        'mu2': mu2, 'std2': std2,
        'delta_mu': delta_mu,
        'l_opt': l_opt, 'sigma_opt': sigma_opt, 'sn_opt': sn_opt
    }

# ── PDE residual computation ──
# PDE: L[u] = v * du/dx - kappa * d2u/dx2 = f(x) = sin(pi*x)
# Residual: R(x) = L[u_pred](x) - f(x)
# For RBF GPR the posterior mean is a linear combination of kernels;
# derivatives are obtained by numerical differentiation of mu on x_star.
def numerical_pde_residual(mu, x, v, kappa):
    """Compute v*du/dx - kappa*d2u/dx2 - f(x) using central differences."""
    dx = x[1] - x[0]   # uniform grid assumed
    # First derivative (central diff, forward/backward at edges)
    du_dx = np.gradient(mu, dx)
    # Second derivative
    d2u_dx2 = np.gradient(du_dx, dx)
    L_mu = v * du_dx - kappa * d2u_dx2
    residual = L_mu - f_rhs(x)
    return residual, L_mu

for kname in ['rbf']:
    r = results[kname]
    res1, Lu1 = numerical_pde_residual(r['mu1'], x_star, v_opt, kap_opt)
    res2, Lu2 = numerical_pde_residual(r['mu2'], x_star, v_opt, kap_opt)
    results[kname]['res1'] = res1
    results[kname]['res2'] = res2
    results[kname]['Lu1']  = Lu1
    results[kname]['Lu2']  = Lu2
    print(f"  RBF PDE residual L2 (GPR):     {np.linalg.norm(res1):.4e}")
    print(f"  RBF PDE residual L2 (Combined):{np.linalg.norm(res2):.4e}")


COLORS = {
    'exact':   '#1a1a2e',
    'gpr':     '#e94560',
    'combined':'#0f3460',
    'dyn_pts': '#f5a623',
    'obs_pts': '#7ed321',
    'col_pts': '#9b59b6',
    'shade2':  '#0f3460',
}

fig = plt.figure(figsize=(28, 38))
fig.patch.set_facecolor('#f8f9fa')

kernel_labels = {'rbf': 'RBF (SE) Kernel'}

for ki, kname in enumerate(['rbf']):
    r   = results[kname]
    col = ki        # 0 or 1
    lbl = kernel_labels[kname]

    # ── Row 1: Full domain view, Pipeline 1 only ──
    ax = fig.add_subplot(5, 2, 2*ki + 1)
    ax.set_facecolor('white')
    ax.fill_between(x_star,
                    r['mu1'] - 2*r['std1'],
                    r['mu1'] + 2*r['std1'],
                    alpha=0.25, color=COLORS['gpr'], label='GPR ±2σ')
    ax.plot(x_star, y_true,  '--', color=COLORS['exact'],  lw=2.5, label='Exact')
    ax.plot(x_star, r['mu1'], '-', color=COLORS['gpr'], lw=2,   label='GPR posterior')
    ax.scatter(x_obs_raw,    y_obs,         s=15, color=COLORS['obs_pts'],
               zorder=5, alpha=0.7, label='Observations')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.08, color='orange', label='Dynamic region')
    ax.set_title(f'{lbl} — Pipeline 1 (GPR only)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(a, b)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Full domain view, Combined ──
    ax = fig.add_subplot(5, 2, 2*ki + 2)
    ax.set_facecolor('white')
    ax.fill_between(x_star,
                    r['mu2'] - 2*r['std2'],
                    r['mu2'] + 2*r['std2'],
                    alpha=0.25, color=COLORS['shade2'], label='Combined ±2σ')
    ax.plot(x_star, y_true,  '--', color=COLORS['exact'],    lw=2.5, label='Exact')
    ax.plot(x_star, r['mu1'],  '-', color=COLORS['gpr'],  lw=1.5,
            alpha=0.5, label='GPR only')
    ax.plot(x_star, r['mu2'],  '-', color=COLORS['combined'],lw=2,   label='Combined posterior')
    ax.scatter(x_dyn, y_dyn, s=25, color=COLORS['dyn_pts'],
               zorder=5, marker='^', label='Dynamic collocations')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.08, color='orange')
    ax.set_title(f'{lbl} — Combined (GPR + GPR correction)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(a, b)
    ax.grid(True, alpha=0.3)

# ── Rows 3-4: Zoom into boundary layer ──
for ki, kname in enumerate(['rbf']):
    r   = results[kname]
    lbl = kernel_labels[kname]
    zoom_mask = x_star >= DYN_LEFT - 0.02

    # Pipeline 1 zoom
    ax = fig.add_subplot(5, 2, 2*ki + 5)
    ax.set_facecolor('white')
    ax.fill_between(x_star[zoom_mask],
                    (r['mu1'] - 2*r['std1'])[zoom_mask],
                    (r['mu1'] + 2*r['std1'])[zoom_mask],
                    alpha=0.3, color=COLORS['gpr'], label='GPR ±2σ')
    ax.plot(x_star[zoom_mask], y_true[zoom_mask],   '--', color=COLORS['exact'],  lw=2.5)
    ax.plot(x_star[zoom_mask], r['mu1'][zoom_mask],  '-', color=COLORS['gpr'], lw=2)
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange')
    ax.set_title(f'{lbl} — Boundary layer zoom (GPR)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.set_xlim(DYN_LEFT - 0.02, b)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Combined zoom
    ax = fig.add_subplot(5, 2, 2*ki + 6)
    ax.set_facecolor('white')
    ax.fill_between(x_star[zoom_mask],
                    (r['mu2'] - 2*r['std2'])[zoom_mask],
                    (r['mu2'] + 2*r['std2'])[zoom_mask],
                    alpha=0.3, color=COLORS['shade2'], label='Combined ±2σ')
    ax.plot(x_star[zoom_mask], y_true[zoom_mask],    '--', color=COLORS['exact'],    lw=2.5)
    ax.plot(x_star[zoom_mask], r['mu1'][zoom_mask],   '-', color=COLORS['gpr'],   lw=1.5,
            alpha=0.5, label='GPR only')
    ax.plot(x_star[zoom_mask], r['mu2'][zoom_mask],   '-', color=COLORS['combined'], lw=2,
            label='Combined')
    ax.scatter(x_dyn, y_dyn, s=40, color=COLORS['dyn_pts'], zorder=5,
               marker='^', label='Dynamic pts')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange')
    ax.set_title(f'{lbl} — Boundary layer zoom (Combined)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.set_xlim(DYN_LEFT - 0.02, b)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


# ── Row 5: PDE residuals L[u_pred] - f(x) ──
for ki, kname in enumerate(['rbf']):
    r   = results[kname]
    lbl = kernel_labels[kname]
    zoom_mask_r = x_star >= DYN_LEFT - 0.02

    # Full-domain PDE residual
    ax = fig.add_subplot(5, 2, 2*ki + 9)
    ax.set_facecolor('white')
    ax.plot(x_star, r['res1'], color=COLORS['gpr'],     lw=2, label='GPR residual')
    ax.plot(x_star, r['res2'], color=COLORS['combined'], lw=2, label='Combined residual')
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange', label='Dynamic region')
    r1_norm = np.linalg.norm(r['res1'])
    r2_norm = np.linalg.norm(r['res2'])
    ax.set_title(
        f"PDE Residual ||R||2: GPR={r1_norm:.3e}, Combined={r2_norm:.3e}",
        fontsize=10, fontweight="bold")
    ax.set_xlabel('x'); ax.set_ylabel('R(x)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Boundary-layer zoom of PDE residual
    ax = fig.add_subplot(5, 2, 2*ki + 10)
    ax.set_facecolor('white')
    ax.plot(x_star[zoom_mask_r], r['res1'][zoom_mask_r], color=COLORS['gpr'],     lw=2, label='GPR residual')
    ax.plot(x_star[zoom_mask_r], r['res2'][zoom_mask_r], color=COLORS['combined'], lw=2, label='Combined residual')
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange', label='Dynamic region')
    ax.set_title(f'{lbl} — PDE Residual Zoom (boundary layer)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('R(x)')
    ax.set_xlim(DYN_LEFT - 0.02, b)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f'1D Advection-Diffusion — Multi-frequency interior forcing  |  '
    f'Pe={Pe:.0f}  |  v={V_TRUE}, κ={KAPPA_TRUE}  |  δ={DELTA:.3f}  |  '
    f'f(x)=sin(3πx)+2sin(7πx)-1.5cos(5πx)+sin(15πx)',
    fontsize=12, fontweight='bold', y=1.002
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.35)
plt.savefig('advdiff_unified.png', dpi=150, bbox_inches='tight')
print("\nMain figure saved.")

# ── Reference solution overview ──
fig0, axes0 = plt.subplots(1, 3, figsize=(18, 4))
fig0.patch.set_facecolor('#f8f9fa')
# forcing
axes0[0].plot(x_star, f_rhs(x_star), color='#9b59b6', lw=2)
axes0[0].set_title('Forcing f(x)', fontweight='bold'); axes0[0].grid(True, alpha=0.3)
axes0[0].set_xlabel('x')
# reference solution
axes0[1].plot(x_star, y_true, color='#1a1a2e', lw=2)
axes0[1].scatter(x_obs_raw, y_obs, s=10, color='#7ed321', alpha=0.6, zorder=5, label='Obs')
axes0[1].scatter(x_dyn, y_dyn, s=30, color='#f5a623', marker='^', zorder=5, label='Dynamic pts')
axes0[1].set_title('Reference solution u(x)', fontweight='bold'); axes0[1].grid(True, alpha=0.3)
axes0[1].set_xlabel('x'); axes0[1].legend(fontsize=8)
# PDE residual of reference (should be ~0 on fine grid)
x_fd_plot = np.linspace(0.01, 0.99, 1000)
cs_res = _cs_ref
du_ref  = cs_res(x_fd_plot, 1)
d2u_ref = cs_res(x_fd_plot, 2)
res_ref = V_TRUE*du_ref - KAPPA_TRUE*d2u_ref - f_rhs(x_fd_plot)
axes0[2].plot(x_fd_plot, res_ref, color='#e94560', lw=1.5)
axes0[2].set_title('FD solution PDE residual (verify: ~0)', fontweight='bold')
axes0[2].grid(True, alpha=0.3); axes0[2].set_xlabel('x')
axes0[2].axhline(0, color='gray', lw=1, ls='--')
plt.tight_layout()
plt.savefig('advdiff_reference.png', dpi=150, bbox_inches='tight')
print("Reference figure saved.")

# ── Error plot ──
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('#f8f9fa')
for ki, kname in enumerate(['rbf']):
    r   = results[kname]
    ax  = axes[ki]
    ax.set_facecolor('white')
    err1 = np.abs(r['mu1'] - y_true)
    err2 = np.abs(r['mu2'] - y_true)
    ax.semilogy(x_star, err1, color=COLORS['gpr'],   lw=2,   label='GPR only')
    ax.semilogy(x_star, err2, color=COLORS['combined'], lw=2,   label='Combined')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange', label='Dynamic region')
    ax.set_title(f'{kernel_labels[kname]} — Absolute Error', fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('|error|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    l2_1 = np.linalg.norm(err1)/np.linalg.norm(y_true)
    l2_2 = np.linalg.norm(err2)/np.linalg.norm(y_true)
    ax.set_title(
        f'{kernel_labels[kname]}\nL2 err: GPR={l2_1:.3f}, Combined={l2_2:.3f}',
        fontweight='bold', fontsize=11)

plt.suptitle('Point-wise absolute error comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('advdiff_errors.png', dpi=150, bbox_inches='tight')
print("Error figure saved.")

# ── GPR correction contribution plot ──
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor('#f8f9fa')
for ki, kname in enumerate(['rbf']):
    r  = results[kname]
    ax = axes[ki]
    ax.set_facecolor('white')
    ax.plot(x_star, r['delta_mu'], color=COLORS['dyn_pts'], lw=2, label='GPR correction Δμ')
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange', label='Dynamic region')
    ax.scatter(x_dyn, np.zeros(len(x_dyn)), s=30, color=COLORS['dyn_pts'],
               zorder=5, marker='^', label='Dynamic pts')
    ax.set_title(f'{kernel_labels[kname]} — GPR Correction Term', fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('Δμ(x)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('GPR correction Δμ = μ_combined − μ_GPR', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('advdiff_correction.png', dpi=150, bbox_inches='tight')
print("Correction figure saved.")

print("\nAll done.")
plt.show()
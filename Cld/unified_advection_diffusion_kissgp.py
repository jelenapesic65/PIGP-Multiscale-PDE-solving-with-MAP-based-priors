"""
Unified 1D High-Pe Advection-Diffusion GP Solver
=================================================
Pipeline 1 (KISSGP):
    - Interior observations
    - Static collocation points (uniform)
    - Boundary condition points
    => Produces global posterior mean + variance

Pipeline 2 (GPR correction):
    - Uses Pipeline 1 posterior as prior mean
    - Dynamic collocation points densely placed near boundary layer
    => Corrects the posterior in the boundary region

Run for both RBF and Matern-5/2 kernels.

High-Pe regime: v=5.0, kappa=0.01 => Pe = v/kappa = 500
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse.linalg import cg
from numpy.linalg import slogdet, solve, cholesky
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
import warnings
warnings.filterwarnings('ignore')


# PROBLEM SETUP  (High-Pe regime)

# True PDE parameters  —  Pe = v / kappa = 500
V_TRUE     = 5.0          # advection velocity
KAPPA_TRUE = 0.01         # diffusion coefficient
Pe         = V_TRUE / KAPPA_TRUE
DELTA      = KAPPA_TRUE / V_TRUE    # boundary layer thickness ≈ 0.002

DOMAIN     = (0.0, 1.0)   # fixed to match u_exact BCs
N_OBS      = 40           # interior observations
N_STATIC_C = 60           # static collocation points
N_INDUCING = 120          # KISSGP inducing grid points
N_DYN      = 30           # dynamic boundary-layer collocation points
N_STAR     = 200          # prediction points
NOISE_OBS  = 1e-3         # observation noise std
NOISE_COL  = 1e-2         # collocation noise std

# Dynamic points: hard-placed densely near x=1 (outflow boundary layer)
# Layer occupies roughly [1 - 5*delta, 1]
DYN_LEFT   = max(0.0, 1.0 - 8 * DELTA)
DYN_RIGHT  = 1.0

np.random.seed(42)
#  EXACT SOLUTION
def u_exact(x, kappa, mu):
    """Exact solution for  mu*u' - kappa*u'' = sin(pi*x),  u(0)=u(1)=0."""
    pi = np.pi
    denom = pi**2 * kappa**2 + mu**2
    A =  kappa / denom
    B = -mu    / (pi * denom)
    r = mu / kappa
    e = np.exp(r)
    C2 = 2*B / (e - 1)
    C1 = -B - C2
    return C1 + C2*np.exp(r*x) + A*np.sin(pi*x) + B*np.cos(pi*x)

def f_rhs(x):
    return np.sin(np.pi * x)

# KERNEL LIBRARY  (RBF and Matern-5/2)

def rbf_k(x, y, l, sigma):
    r2 = (x[:, None] - y[None, :])**2
    return sigma**2 * np.exp(-0.5 * r2 / l**2)

def matern52_k(x, y, l, sigma):
    r  = np.abs(x[:, None] - y[None, :])
    s  = np.sqrt(5) * r / l
    return sigma**2 * (1 + s + s**2/3) * np.exp(-s)

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

def matern52_derivs(x, y, l, sigma):
    """Exact derivatives of Matern-5/2 kernel (1D)."""
    r_raw = x[:, None] - y[None, :]
    r     = np.abs(r_raw)
    sgn   = np.sign(r_raw)            
    s5    = np.sqrt(5)
    s     = s5 * r / l
    base  = sigma**2 * np.exp(-s)

    k     = base * (1 + s + s**2/3)

    dkdr  = base * (-s5/l) * (s + s**2/3)   
  
    dkdr  = -(5*r/(3*l**2)) * sigma**2 * (1 + s5*r/l) * np.exp(-s)

    k_x   = dkdr * sgn
    k_y   = -k_x

    d2kdr2 = sigma**2 * np.exp(-s) * (5/(3*l**2)) * (s5*s*r/l - 1 - s)

    k_xx  = d2kdr2 * sgn**2    # sgn^2 = 1 except at r=0
    k_yy  = k_xx
    k_xy  = -k_xx

    # Third and fourth derivatives approximate via finite differences for Matern
    # finite diff is stable for l not tiny
    eps   = 1e-5
    def _k(xi, yi): return matern52_k(xi, yi, l, sigma)
    xv = x[:, None] * np.ones_like(y[None, :])
    yv = np.ones_like(x[:, None]) * y[None, :]
    xarr = xv.ravel()
    yarr = yv.ravel()
    def ksc(xi, yi):
        return matern52_k(np.array([xi]), np.array([yi]), l, sigma)[0, 0]

    shape = (len(x), len(y))

    k_xxy  = np.zeros(shape)
    k_xyy  = np.zeros(shape)
    k_xxyy = np.zeros(shape)
    for i in range(len(x)):
        for j in range(len(y)):
            xi, yj = x[i], y[j]
            # d3k/dx2 dy  via central diff in y of d2k/dx2
            def d2kdx2(yi):
                return (ksc(xi+eps, yi) - 2*ksc(xi, yi) + ksc(xi-eps, yi)) / eps**2
            k_xxy[i,j] = (d2kdx2(yj+eps) - d2kdx2(yj-eps)) / (2*eps)
            # d3k/dx dy2  via central diff in x of d2k/dy2
            def d2kdy2(xi2):
                return (ksc(xi2, yj+eps) - 2*ksc(xi2, yj) + ksc(xi2, yj-eps)) / eps**2
            k_xyy[i,j] = (d2kdy2(xi+eps) - d2kdy2(xi-eps)) / (2*eps)
            # d4k/dx2 dy2
            k_xxyy[i,j] = (d2kdx2(yj+eps) - 2*d2kdx2(yj) + d2kdx2(yj-eps)) / eps**2

    return k, k_x, k_y, k_xx, k_yy, k_xy, k_xxy, k_xyy, k_xxyy


def get_derivs(kernel_name, x, y, l, sigma):
    if kernel_name == 'rbf':
        return rbf_derivs(x, y, l, sigma)
    else:
        return matern52_derivs(x, y, l, sigma)

def get_kernel(kernel_name, x, y, l, sigma):
    if kernel_name == 'rbf':
        return rbf_k(x, y, l, sigma)
    else:
        return matern52_k(x, y, l, sigma)


def interp_w0(x, U, h):
    W = np.zeros(len(U))
    j = np.clip(np.searchsorted(U, x) - 1, 0, len(U)-2)
    t = (x - U[j]) / h
    W[j]   = 1 - t
    W[j+1] = t
    return W

def interp_w1(x, U, h):
    W = np.zeros(len(U))
    j = np.clip(np.searchsorted(U, x) - 1, 0, len(U)-2)
    W[j]   = -1/h
    W[j+1] =  1/h
    return W

def interp_w2(x, U, h):
    W = np.zeros(len(U))
    j = np.clip(np.searchsorted(U, x) - 1, 0, len(U)-2)
    if j > 0:
        W[j-1] =  1/h**2
    W[j]   = -2/h**2
    if j+1 < len(U)-1:
        W[j+1] =  1/h**2
    return W

#  KISSGP PIPELINE  (Pipeline 1)

def build_KISSGP_system(U, x_obs, y_obs, x_col, y_col,
                         l, sigma, sigma_n, v, kappa,
                         kernel_name='rbf'):
    """
      Rows: [BC_left, BC_right, obs_interior, collocation]
      Cols: inducing grid U

    Returns: Kuu, W, y_all
    """
    m  = len(U)
    h  = U[1] - U[0]
    a_, b_ = U[0], U[-1]

    Kuu = get_kernel(kernel_name, U, U, l, sigma)

    rows = []
    # BC at x=0
    w = interp_w0(a_, U, h); rows.append(w)
    # BC at x=1
    w = interp_w0(b_, U, h); rows.append(w)
    # Interior observations
    for xi in x_obs:
        rows.append(interp_w0(xi, U, h))
    # Static collocation: PDE operator  v*d/dx - kappa*d2/dx2
    for xi in x_col:
        w1 = interp_w1(xi, U, h)
        w2 = interp_w2(xi, U, h)
        rows.append(v*w1 - kappa*w2)

    W     = np.vstack(rows)
    y_all = np.concatenate([[0.0, 0.0], y_obs, y_col])
    return Kuu, W, y_all


def kissgp_posterior(U, x_obs, y_obs, x_col, y_col,
                     x_star, l, sigma, sigma_n, v, kappa,
                     kernel_name='rbf'):

    Kuu, W, y_all = build_KISSGP_system(
        U, x_obs, y_obs, x_col, y_col,
        l, sigma, sigma_n, v, kappa, kernel_name)

    n_rows = W.shape[0]
    K      = W @ Kuu @ W.T + sigma_n**2 * np.eye(n_rows)
    K     += 1e-8 * np.eye(n_rows)

    h = U[1] - U[0]
    W_star = np.vstack([interp_w0(xi, U, h) for xi in x_star])

    alpha, info = cg(K, y_all, maxiter=10000, tol=1e-10)

    mu_star = W_star @ Kuu @ W.T @ alpha

 
    K_star_rows = W_star @ Kuu @ W.T          # (n_star, n_rows)
    K_star_diag = np.diag(W_star @ Kuu @ W_star.T)

    V_mat = np.zeros_like(K_star_rows.T)
    for i in range(K_star_rows.shape[0]):
        v_i, _ = cg(K, K_star_rows[i], maxiter=10000, tol=1e-10)
        V_mat[:, i] = v_i
    var_star = K_star_diag - np.einsum('ij,ji->i', K_star_rows, V_mat)
    var_star = np.maximum(var_star, 0.0)

    return mu_star, var_star, Kuu, W, K, alpha, W_star


def kissgp_nlml(theta, U, x_obs, y_obs, x_col, y_col, kernel_name):
    l, sigma, log_sn, v, kap = theta
    sn = np.exp(log_sn)
    Kuu, W, y_all = build_KISSGP_system(
        U, x_obs, y_obs, x_col, y_col,
        l, sigma, sn, v, kap, kernel_name)
    n = W.shape[0]
    K = W @ Kuu @ W.T + sn**2 * np.eye(n) + 1e-8*np.eye(n)
    sign, ld = slogdet(K)
    if sign <= 0:
        return 1e10
    alpha = solve(K, y_all)
    nlml  = 0.5 * y_all @ alpha + 0.5 * ld

    nlml -= -0.5*((np.log(l)   - np.log(0.1))/0.8)**2
    nlml -= -0.5*((np.log(sigma)- np.log(1.0))/0.5)**2
    nlml -= -0.5*((log_sn      - np.log(1e-3))/1.0)**2
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
    RBF) do not degrade the already-good KISSGP solution.

    Returns corrected mean and variance at x_star.
    """
    r_obs = y_dyn - mu_prior_dyn


    # Build kernel matrices for residual GP
    Kdd = get_kernel(kernel_name, x_dyn, x_dyn, l_loc, sigma_loc)
    Kdd += sigma_n_loc**2 + np.mean(var_prior_dyn) + 1e-8 * np.eye(len(x_dyn))

    Ksd = get_kernel(kernel_name, x_star, x_dyn, l_loc, sigma_loc)
    Kss_diag = sigma_loc**2 * np.ones(len(x_star))

    L     = cho_factor(Kdd)
    alpha = cho_solve(L, r_obs)

    delta_mu  = Ksd @ alpha
    v_        = cho_solve(L, Ksd.T)
    delta_var = Kss_diag - np.einsum('ij,ji->i', Ksd, v_)
    delta_var = np.maximum(delta_var, 0.0)

    # ── Smooth taper: correction fades to zero outside the dynamic region.
    #    Use a logistic window with width ~ l_loc so the transition is smooth
    #    rather than abrupt.  This prevents RBF from oscillating in the bulk.
    width = max(l_loc, 3 * (dyn_right - dyn_left) / len(x_dyn))
    taper = 1.0 / (1.0 + np.exp(-(x_star - dyn_left + width) / (width/4))) \
          * 1.0 / (1.0 + np.exp( (x_star - dyn_right - width) / (width/4)))
    # taper ≈ 1 inside dynamic region, ≈ 0 well outside it
    delta_mu_tapered = delta_mu * taper

    mu_combined  = mu_prior_star + delta_mu_tapered
    var_combined = var_prior_star + delta_var * taper

    return mu_combined, var_combined, delta_mu_tapered, delta_var



a, b = DOMAIN

x_obs_raw = np.sort(np.random.uniform(a, 1-10*DELTA, N_OBS))
y_obs      = u_exact(x_obs_raw, KAPPA_TRUE, V_TRUE) + np.random.normal(0, NOISE_OBS, N_OBS)


x_col_static = np.linspace(a, b, N_STATIC_C+2)[1:-1]   # exclude endpoints
y_col_static  = f_rhs(x_col_static) + np.random.normal(0, NOISE_COL, N_STATIC_C)

x_dyn = np.sort(np.random.uniform(DYN_LEFT, DYN_RIGHT, N_DYN))
y_dyn = u_exact(x_dyn, KAPPA_TRUE, V_TRUE) + np.random.normal(0, NOISE_OBS, N_DYN)

# Inducing grid (uniform over [0,1])
U = np.linspace(a, b, N_INDUCING)
h = U[1] - U[0]

# Prediction grid
x_star = np.linspace(a, b, N_STAR)
y_true = u_exact(x_star, KAPPA_TRUE, V_TRUE)

print(f"Problem setup:")
print(f"  Pe = {Pe:.0f},  delta = {DELTA:.4f}")
print(f"  Boundary layer region: [{DYN_LEFT:.4f}, {DYN_RIGHT:.3f}]")
print(f"  n_obs={N_OBS}, n_static_col={N_STATIC_C}, n_dyn={N_DYN}, n_inducing={N_INDUCING}")


THETA0_KISSGP = np.array([
    0.15,          # l  
    1.0,           # sigma
    np.log(1e-3),  # log sigma_n
    V_TRUE,        # v  
    KAPPA_TRUE    # kappa 
])

BOUNDS_KISSGP = [
    (0.02, 0.5),        # l
    (0.1,  5.0),        # sigma
    (-10,  -1),         # log sigma_n
    (V_TRUE, V_TRUE),   # v fixed
    (KAPPA_TRUE, KAPPA_TRUE)  # kappa fixed
]

# Length scale ~ delta (boundary layer thickness)
L_LOCAL    = max(3 * DELTA, 0.005)   # at least a few grid spacings
SIGMA_LOCAL = 0.5
SN_LOCAL    = NOISE_OBS


results = {}

for kname in ['rbf', 'matern52']:
    print(f"\n{'='*60}")
    print(f"  Kernel: {kname.upper()}")
    print('='*60)

    # ── Optimize KISSGP hyperparameters ──
    print("  Optimizing KISSGP hyperparameters...")
    res = minimize(
        kissgp_nlml,
        THETA0_KISSGP,
        args=(U, x_obs_raw, y_obs, x_col_static, y_col_static, kname),
        method='L-BFGS-B',
        bounds=BOUNDS_KISSGP,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    l_opt, sigma_opt, log_sn_opt, v_opt, kap_opt = res.x
    sn_opt = np.exp(log_sn_opt)
    print(f"  l={l_opt:.4f}, sigma={sigma_opt:.4f}, sn={sn_opt:.2e}, "
          f"v={v_opt:.3f}, kappa={kap_opt:.4f}")

    # ── Pipeline 1: KISSGP posterior ──
    print("  Running KISSGP pipeline...")
    (mu1, var1, Kuu, W, K_sys, alpha_sys, W_star) = kissgp_posterior(
        U, x_obs_raw, y_obs, x_col_static, y_col_static,
        x_star, l_opt, sigma_opt, sn_opt, v_opt, kap_opt, kname)
    std1 = np.sqrt(var1)
    print(f"  KISSGP L2 error: {np.linalg.norm(mu1 - y_true)/np.linalg.norm(y_true):.4f}")

    # ── Get KISSGP posterior at dynamic points (as prior for Pipeline 2) ──
    (mu1_dyn, var1_dyn, _, _, _, _, _) = kissgp_posterior(
        U, x_obs_raw, y_obs, x_col_static, y_col_static,
        x_dyn, l_opt, sigma_opt, sn_opt, v_opt, kap_opt, kname)

    # ── Pipeline 2: GPR correction ──
    print("  Running GPR correction pipeline...")
    # local kernel uses same type but with local length scale
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


COLORS = {
    'exact':   '#1a1a2e',
    'kissgp':  '#e94560',
    'combined':'#0f3460',
    'dyn_pts': '#f5a623',
    'obs_pts': '#7ed321',
    'col_pts': '#9b59b6',
    'shade1':  '#e94560',
    'shade2':  '#0f3460',
}

fig = plt.figure(figsize=(28, 30))
fig.patch.set_facecolor('#f8f9fa')

kernel_labels = {'rbf': 'RBF (SE) Kernel', 'matern52': 'Matérn-5/2 Kernel'}

for ki, kname in enumerate(['rbf', 'matern52']):
    r   = results[kname]
    col = ki        # 0 or 1
    lbl = kernel_labels[kname]

    # ── Row 1: Full domain view, Pipeline 1 only ──
    ax = fig.add_subplot(4, 2, 2*ki + 1)
    ax.set_facecolor('white')
    ax.fill_between(x_star,
                    r['mu1'] - 2*r['std1'],
                    r['mu1'] + 2*r['std1'],
                    alpha=0.25, color=COLORS['shade1'], label='KISSGP ±2σ')
    ax.plot(x_star, y_true,  '--', color=COLORS['exact'],  lw=2.5, label='Exact')
    ax.plot(x_star, r['mu1'], '-', color=COLORS['kissgp'], lw=2,   label='KISSGP posterior')
    ax.scatter(x_obs_raw,    y_obs,         s=15, color=COLORS['obs_pts'],
               zorder=5, alpha=0.7, label='Observations')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.08, color='orange', label='Dynamic region')
    ax.set_title(f'{lbl} — Pipeline 1 (KISSGP only)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(a, b)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Full domain view, Combined ──
    ax = fig.add_subplot(4, 2, 2*ki + 2)
    ax.set_facecolor('white')
    ax.fill_between(x_star,
                    r['mu2'] - 2*r['std2'],
                    r['mu2'] + 2*r['std2'],
                    alpha=0.25, color=COLORS['shade2'], label='Combined ±2σ')
    ax.plot(x_star, y_true,  '--', color=COLORS['exact'],    lw=2.5, label='Exact')
    ax.plot(x_star, r['mu1'],  '-', color=COLORS['kissgp'],  lw=1.5,
            alpha=0.5, label='KISSGP only')
    ax.plot(x_star, r['mu2'],  '-', color=COLORS['combined'],lw=2,   label='Combined posterior')
    ax.scatter(x_dyn, y_dyn, s=25, color=COLORS['dyn_pts'],
               zorder=5, marker='^', label='Dynamic collocations')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.08, color='orange')
    ax.set_title(f'{lbl} — Combined (KISSGP + GPR correction)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(a, b)
    ax.grid(True, alpha=0.3)

# ── Rows 3-4: Zoom into boundary layer ──
for ki, kname in enumerate(['rbf', 'matern52']):
    r   = results[kname]
    lbl = kernel_labels[kname]
    zoom_mask = x_star >= DYN_LEFT - 0.02

    # Pipeline 1 zoom
    ax = fig.add_subplot(4, 2, 2*ki + 5)
    ax.set_facecolor('white')
    ax.fill_between(x_star[zoom_mask],
                    (r['mu1'] - 2*r['std1'])[zoom_mask],
                    (r['mu1'] + 2*r['std1'])[zoom_mask],
                    alpha=0.3, color=COLORS['shade1'], label='KISSGP ±2σ')
    ax.plot(x_star[zoom_mask], y_true[zoom_mask],   '--', color=COLORS['exact'],  lw=2.5)
    ax.plot(x_star[zoom_mask], r['mu1'][zoom_mask],  '-', color=COLORS['kissgp'], lw=2)
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange')
    ax.set_title(f'{lbl} — Boundary layer zoom (KISSGP)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u(x)')
    ax.set_xlim(DYN_LEFT - 0.02, b)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Combined zoom
    ax = fig.add_subplot(4, 2, 2*ki + 6)
    ax.set_facecolor('white')
    ax.fill_between(x_star[zoom_mask],
                    (r['mu2'] - 2*r['std2'])[zoom_mask],
                    (r['mu2'] + 2*r['std2'])[zoom_mask],
                    alpha=0.3, color=COLORS['shade2'], label='Combined ±2σ')
    ax.plot(x_star[zoom_mask], y_true[zoom_mask],    '--', color=COLORS['exact'],    lw=2.5)
    ax.plot(x_star[zoom_mask], r['mu1'][zoom_mask],   '-', color=COLORS['kissgp'],   lw=1.5,
            alpha=0.5, label='KISSGP only')
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

plt.suptitle(
    f'1D Advection-Diffusion  |  Pe = {Pe:.0f}  |  '
    f'v={V_TRUE}, κ={KAPPA_TRUE}  |  δ≈{DELTA:.4f}',
    fontsize=14, fontweight='bold', y=1.002
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.35)
plt.savefig('advdiff_unified.png', dpi=150, bbox_inches='tight')
print("\nMain figure saved.")

# ── Error plot ──
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('#f8f9fa')
for ki, kname in enumerate(['rbf', 'matern52']):
    r   = results[kname]
    ax  = axes[ki]
    ax.set_facecolor('white')
    err1 = np.abs(r['mu1'] - y_true)
    err2 = np.abs(r['mu2'] - y_true)
    ax.semilogy(x_star, err1, color=COLORS['kissgp'],   lw=2,   label='KISSGP only')
    ax.semilogy(x_star, err2, color=COLORS['combined'], lw=2,   label='Combined')
    ax.axvspan(DYN_LEFT, DYN_RIGHT, alpha=0.1, color='orange', label='Dynamic region')
    ax.set_title(f'{kernel_labels[kname]} — Absolute Error', fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('|error|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    l2_1 = np.linalg.norm(err1)/np.linalg.norm(y_true)
    l2_2 = np.linalg.norm(err2)/np.linalg.norm(y_true)
    ax.set_title(
        f'{kernel_labels[kname]}\nL2 err: KISSGP={l2_1:.3f}, Combined={l2_2:.3f}',
        fontweight='bold', fontsize=11)

plt.suptitle('Point-wise absolute error comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('advdiff_errors.png', dpi=150, bbox_inches='tight')
print("Error figure saved.")

# ── GPR correction contribution plot ──
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor('#f8f9fa')
for ki, kname in enumerate(['rbf', 'matern52']):
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

plt.suptitle('GPR correction Δμ = μ_combined − μ_KISSGP', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('advdiff_correction.png', dpi=150, bbox_inches='tight')
print("Correction figure saved.")

print("\nAll done.")
plt.show()

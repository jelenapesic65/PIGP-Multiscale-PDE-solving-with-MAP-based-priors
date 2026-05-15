"""


PDE:  v_x * du/dx + v_y * du/dy - kappa * (d2u/dx2 + d2u/dy2) = f(x,y)
BCs:  u = 0  on all four edges

Advection is purely in +x direction (v_x = V_TRUE, v_y = 0), so the
boundary layer forms at the outflow edge x = 1.

Exact solution is separable:
    u(x,y) = u_1d(x) * sin(pi*y)
where u_1d solves  v*u' - kappa*u'' = sin(pi*x) with homogeneous BCs,
and f(x,y) = sin(pi*x)*sin(pi*y) + kappa*pi^2 * u_1d(x)*sin(pi*y).




Run for RBF kernel only (Matern-5/2 2D mixed derivatives require either
symbolic or expensive finite-difference; easily extendable).

HighPe regime: v=5.0, kappa=0.01 => Pe = 500
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import plotly.io as pio
pio.renderers.default = "browser"
# ─────────────────────────────────────────────
#  PROBLEM SETUP
# ─────────────────────────────────────────────

V_TRUE     = 5.0
KAPPA_TRUE = 0.01
Pe         = V_TRUE / KAPPA_TRUE
DELTA      = KAPPA_TRUE / V_TRUE      # boundary layer thickness ~ 0.002

# Point counts
N_OBS      = 80     # scattered interior observations
N_STATIC_C = 120    # static interior collocation points
N_BC       = 60     # boundary condition points (split across 4 edges)
N_DYN      = 70     # dynamic points near x=1 boundary layer
N_STAR_1D  = 40     # prediction grid per axis  => N_STAR_1D^2 total

NOISE_OBS  = 1e-3
NOISE_COL  = 1e-3

# Dynamic region: strip near the outflow boundary x=1
DYN_X_LEFT = max(0.0, 1.0 - 15 * DELTA)
DYN_X_RIGHT = 1.0

np.random.seed(42)

# ─────────────────────────────────────────────
#  EXACT SOLUTION  (separable)
# ─────────────────────────────────────────────

def u1d_exact(x, kappa, v):
    """
    Exact solution of  v*U' - kappa*U'' + kappa*pi^2*U = sin(pi*x),
    U(0)=U(1)=0.
    """
    pi    = np.pi
    disc  = np.sqrt(v**2 + 4.0*kappa**2*pi**2)
    r1    = (v - disc) / (2.0*kappa)
    r2    = (v + disc) / (2.0*kappa)
 
    denom = 4.0*kappa**2*pi**2 + v**2
    A     =  2.0*kappa / denom
    B     = -v         / (pi * denom)
 
    e1, e2 = np.exp(r1), np.exp(r2)
    C2    = B * (1.0 + e1) / (e2 - e1)
    C1    = -B - C2
 
    return C1*np.exp(r1*x) + C2*np.exp(r2*x) + A*np.sin(pi*x) + B*np.cos(pi*x)
 
 
def u_exact_2d(xy, kappa, v):
    """u(x,y) = U(x)*sin(pi*y),  xy: (N,2)."""
    return u1d_exact(xy[:, 0], kappa, v) * np.sin(np.pi * xy[:, 1])
 
 
def f_rhs_2d(xy, kappa, v):
    """
    Prescribed forcing  f(x,y) = sin(pi*x)*sin(pi*y).
    Clean and parameter-free.
    """
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

# ─────────────────────────────────────────────
#  2D PRODUCT-RBF KERNEL AND ALL NEEDED DERIVATIVES
#
#  k(p,q) = sigma^2 * exp(-rx^2/(2*lx^2)) * exp(-ry^2/(2*ly^2))
#          = sigma^2 * Kx * Ky
#
#  where p=(x1,y1), q=(x2,y2), rx=x1-x2, ry=y1-y2.
#
#  The 2D PDE operator is  L = v*d/dx - kappa*(d^2/dx^2 + d^2/dy^2).
#
#  Required kernel blocks:
#    K_vv : k(p,q)
#    K_vc : L_q k(p,q) = v*k_x2 - kappa*(k_x2x2 + k_y2y2)
#    K_cc : L_p L_q k(p,q)
#         = (v*d/dx1 - kappa*(d^2/dx1^2 + d^2/dy1^2))
#           (v*k_x2 - kappa*(k_x2x2 + k_y2y2))
#
#  All derivatives are analytic for the product RBF.
# ─────────────────────────────────────────────

def _rbf_factors(p, q, lx, ly, sigma):
    """
    Return scalar factors needed to build all required derivative blocks.

    p : (M,2)  first set of points
    q : (N,2)  second set of points

    Returns arrays of shape (M,N):
        Kx, Ky, rx, ry  (and derived quantities)
    """
    rx  = p[:, 0:1] - q[None, :, 0]   # (M,N)  -- note: p[:,0:1] is (M,1)
    ry  = p[:, 1:2] - q[None, :, 1]

    # Fix indexing: p is (M,2), so p[:,0:1] shape=(M,1); q[:,0] shape=(N,)
    # Use explicit broadcasting
    rx  = p[:, 0][:, None] - q[:, 0][None, :]   # (M,N)
    ry  = p[:, 1][:, None] - q[:, 1][None, :]   # (M,N)

    lx2, ly2 = lx**2, ly**2

    Kx  = np.exp(-0.5 * rx**2 / lx2)
    Ky  = np.exp(-0.5 * ry**2 / ly2)
    K   = sigma**2 * Kx * Ky

    return K, Kx, Ky, rx, ry, lx2, ly2


def k2d(p, q, lx, ly, sigma):
    """Plain 2D product-RBF kernel, shape (M,N)."""
    K, *_ = _rbf_factors(p, q, lx, ly, sigma)
    return K


def k2d_L2(p, q, lx, ly, sigma, v, kappa):
    """
    L applied to second argument:  L_q k(p,q)
      = v * dk/dx2 - kappa*(d^2k/dx2^2 + d^2k/dy2^2)

    Note: dk/dx2 = -dk/dx1 (shift symmetry), so dk/dx2 = +rx/lx^2 * K
          d^2k/dx2^2 = d^2k/dx1^2 = (rx^2/lx^4 - 1/lx^2) * K
          d^2k/dy2^2 = (ry^2/ly^4 - 1/ly^2) * K
    """
    K, Kx, Ky, rx, ry, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)

    dK_dx2   =  (rx / lx2) * K
    d2K_dx22 = (rx**2 / lx2**2 - 1.0 / lx2) * K
    d2K_dy22 = (ry**2 / ly2**2 - 1.0 / ly2) * K

    return v * dK_dx2 - kappa * (d2K_dx22 + d2K_dy22)


def k2d_L1L2(p, q, lx, ly, sigma, v, kappa):
    """
    L_p L_q k(p,q) :  L applied to BOTH arguments.

    L_p = v*d/dx1 - kappa*(d^2/dx1^2 + d^2/dy1^2)
    L_q = v*d/dx2 - kappa*(d^2/dx2^2 + d^2/dy2^2)

    Expand:  L_p (L_q k) where L_q k is computed above.
    Let  A = dK/dx2,  B = d2K/dx22,  C = d2K/dy22
    Then L_q k = v*A - kappa*(B+C).

    d/dx1(A) = d/dx1(rx/lx2 * K) = (1/lx2)*K + (rx/lx2)*(-rx/lx2)*K
                                   = (1/lx2 - rx^2/lx2^2)*K  ... wait:
    Actually d/dx1 means x1 changes; rx = x1-x2, so d(rx)/dx1=1.
      dK/dx1 = (-rx/lx2)*K   (standard RBF derivative)
      dA/dx1 = d/dx1(rx/lx2 * K) = (1/lx2)*K + (rx/lx2)*(-rx/lx2)*K
             = (1/lx2 - rx^2/lx2^2)*K = -d2K/dx1^2_but_sign ...

    Let us just define the four scalar functions in terms of K:
      ax = -rx/lx2           (dK/dx1 = ax*K)
      axx = rx^2/lx2^2 - 1/lx2  (d2K/dx1^2 = axx*K)
      ayy = ry^2/ly2^2 - 1/ly2  (d2K/dy1^2 = ayy*K)
      bx  = +rx/lx2           (dK/dx2 = bx*K)
      bxx = axx               (d2K/dx2^2 = axx*K, same formula)
      byy = ayy
      axbx = ax*bx = -rx^2/lx2^2  (d2K/dx1dx2 = axbx*K ... NO, it's (ax+bx derivative))

    Clean derivation:
      K = sigma^2 * exp(-rx^2/2lx^2 - ry^2/2ly^2)
      Let ex = rx/lx2, ey = ry/ly2

      dK/dx1 = -ex*K
      dK/dx2 = +ex*K
      d2K/dx1^2 = (ex^2 - 1/lx2)*K
      d2K/dy1^2 = (ey^2 - 1/ly2)*K
      d2K/dx2^2 = (ex^2 - 1/lx2)*K   [same as d2/dx1^2]
      d2K/dy2^2 = (ey^2 - 1/ly2)*K
      d3K/dx1dx2^2 = d/dx1[(ex^2-1/lx2)*K] = [2*ex*(1/lx2) + (ex^2-1/lx2)*(-ex)]*K
                   = ex*(2/lx2 - ex^2 + 1/lx2)*K = ex*(3/lx2 - ex^2)*K   ... let's redo:
      d/dx1[(ex^2-1/lx2)*K] where ex=rx/lx2, d(ex)/dx1=1/lx2:
         = (2*ex*(1/lx2))*K + (ex^2-1/lx2)*(-ex*K)
         = K*ex*(2/lx2 - ex^2 + 1/lx2)   NO:
         = K*(2*ex/lx2 - ex^3 + ex/lx2)  ... let me be careful:
         = 2*(1/lx2)*ex*K + (ex^2-1/lx2)*(-ex)*K
         = K*ex*(2/lx2 - ex^2 + 1/lx2)
         Hmm that gives K*ex*(3/lx2 - ex^2)... let me just use alpha notation.

    We use scalar multipliers (all shape (M,N)):
    """
    K, Kx, Ky, rx, ry, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)

    ex  = rx / lx2   # (M,N)
    ey  = ry / ly2

    # First-order multipliers (times K)
    #   dK/dx1 = -ex*K,   dK/dx2 = +ex*K
    #   dK/dy1 = -ey*K,   dK/dy2 = +ey*K

    # Second-order multipliers
    axx  = ex**2 - 1.0/lx2    # d2K/dx1^2 / K  (same for dx2^2)
    ayy  = ey**2 - 1.0/ly2    # d2K/dy1^2 / K

    # L_q k = (v*ex - kappa*(axx + ayy)) * K  =: Lq * K
    Lq = v * ex - kappa * (axx + ayy)

    # L_p (Lq*K):
    #   d/dx1(Lq*K) = (dLq/dx1)*K + Lq*(dK/dx1)
    #               = (dLq/dx1 - Lq*ex)*K
    #
    #   d2/dx1^2(Lq*K) = (d2Lq/dx1^2 - 2*(dLq/dx1)*ex + Lq*(1/lx2 - ex^2)... let's expand:
    #   Let F = Lq*K. Then:
    #   dF/dx1   = (dLq/dx1)*K + Lq*(-ex*K) = (dLq/dx1 - ex*Lq)*K
    #   d2F/dx1^2= (d2Lq/dx1^2 - ex*(dLq/dx1) - (1/lx2)*Lq ... careful:
    #   d2F/dx1^2= d/dx1[(dLq/dx1 - ex*Lq)*K]
    #            = (d2Lq/dx1^2 - (1/lx2)*Lq - ex*(dLq/dx1))*K
    #              + (dLq/dx1 - ex*Lq)*(-ex*K)
    #            = (d2Lq/dx1^2 - (1/lx2)*Lq - 2*ex*(dLq/dx1) + ex^2*Lq)*K
    #   d2F/dy1^2= (d2Lq/dy1^2 - (1/ly2)*Lq - 2*ey*(dLq/dy1) + ey^2*Lq)*K
    #
    # Compute derivatives of Lq w.r.t. x1, y1:
    #   Lq = v*ex - kappa*(ex^2 - 1/lx2 + ey^2 - 1/ly2)
    #   dLq/dx1 = v*(1/lx2) - kappa*(2*ex*(1/lx2))
    #           = (v - 2*kappa*ex)/lx2
    #   d2Lq/dx1^2 = (0 - 2*kappa*(1/lx2))/lx2 = -2*kappa/lx2^2
    #   dLq/dy1 = -kappa*(2*ey*(1/ly2)) = -2*kappa*ey/ly2
    #   d2Lq/dy1^2 = -2*kappa/ly2^2

    dLq_dx1  = (v - 2*kappa*ex) / lx2
    d2Lq_dx1 = -2*kappa / lx2**2
    dLq_dy1  = -2*kappa*ey / ly2
    d2Lq_dy1 = -2*kappa / ly2**2

    # L_p(Lq*K) = v*dF/dx1 - kappa*(d2F/dx1^2 + d2F/dy1^2)
    dF_dx1   = (dLq_dx1 - ex*Lq) * K
    d2F_dx1  = (d2Lq_dx1 - 1.0/lx2 * Lq - 2*ex*dLq_dx1 + ex**2*Lq) * K
    d2F_dy1  = (d2Lq_dy1 - 1.0/ly2 * Lq - 2*ey*dLq_dy1 + ey**2*Lq) * K

    return v*dF_dx1 - kappa*(d2F_dx1 + d2F_dy1)


# ─────────────────────────────────────────────
#  PIPELINE 1: GLOBAL GPR
# ─────────────────────────────────────────────

def build_gpr_system_2d(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                         lx, ly, sigma, sigma_n, v, kappa):
    """
    Assemble the full GPR kernel matrix for the 2D problem.

    Training rows (in order):
      [BC points | interior observations | collocation points]

    BC and observation rows → plain kernel k2d.
    Collocation rows        → L_q applied kernel k2d_L2.
    """
    # Stack value-type points: BC + obs
    xy_val = np.vstack([xy_bc, xy_obs])
    u_val  = np.concatenate([u_bc, u_obs])

    n_v  = len(xy_val)
    n_c  = len(xy_col)
    n    = n_v + n_c

    y_train = np.concatenate([u_val, f_col])

    # Block (val, val)
    K_vv = k2d(xy_val, xy_val, lx, ly, sigma)

    # Block (val, col): L applied to second arg
    K_vc = k2d_L2(xy_val, xy_col, lx, ly, sigma, v, kappa)

    # Block (col, col): L applied to both args
    K_cc = k2d_L1L2(xy_col, xy_col, lx, ly, sigma, v, kappa)

    K_train = np.zeros((n, n))
    K_train[:n_v, :n_v] = K_vv
    K_train[:n_v, n_v:] = K_vc
    K_train[n_v:, :n_v] = K_vc.T
    K_train[n_v:, n_v:] = K_cc

    K_train += sigma_n**2 * np.eye(n)
    K_train += 1e-8 * np.eye(n)

    return xy_val, y_train, K_train, n_v


def gpr_posterior_2d(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                     xy_star, lx, ly, sigma, sigma_n, v, kappa):
    """Standard GPR posterior mean and diagonal variance at xy_star."""
    xy_val, y_train, K_train, n_v = build_gpr_system_2d(
        xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
        lx, ly, sigma, sigma_n, v, kappa)

    xy_col_ = xy_star   # alias for clarity below
    n_c_train = len(y_train) - n_v

    # Cholesky solve
    L_fac = cho_factor(K_train)
    alpha = cho_solve(L_fac, y_train)

    # Recover xy_col used in training (last n_c_train rows of y_train correspond to it)
    # We need to rebuild the collocation input
    # (passed in as xy_col to this function)
    xy_col_train = xy_col   # the collocation points used to build the system

    # Cross-covariance: k(x_star, x_train)
    #   value columns  -> k2d(xy_star, xy_val)
    #   operator cols  -> k2d_L2(xy_star, xy_col_train)  [L on second arg]
    Ks_val = k2d(xy_star, xy_val, lx, ly, sigma)
    Ks_col = k2d_L2(xy_star, xy_col_train, lx, ly, sigma, v, kappa)
    K_star = np.hstack([Ks_val, Ks_col])   # (n_star, n_train)

    mu_star = K_star @ alpha

    # Diagonal posterior variance
    Kss_diag = sigma**2 * np.ones(len(xy_star))
    V = cho_solve(L_fac, K_star.T)
    var_star = Kss_diag - np.einsum('ij,ji->i', K_star, V)
    var_star = np.maximum(var_star, 0.0)

    return mu_star, var_star, K_train, alpha


def gpr_nlml_2d(theta, xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc):
    lx, ly, sigma, log_sn = theta
    sn = np.exp(log_sn)
    xy_val, y_train, K_train, n_v = build_gpr_system_2d(
        xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
        lx, ly, sigma, sn, V_TRUE, KAPPA_TRUE)
    sign, ld = slogdet(K_train)
    if sign <= 0:
        return 1e10
    alpha = solve(K_train, y_train)
    nlml  = 0.5 * y_train @ alpha + 0.5 * ld
    # Log-normal priors
    nlml -= -0.5 * ((np.log(lx)    - np.log(0.15)) / 0.8)**2
    nlml -= -0.5 * ((np.log(ly)    - np.log(0.15)) / 0.8)**2
    nlml -= -0.5 * ((np.log(sigma) - np.log(1.0))  / 0.5)**2
    nlml -= -0.5 * ((log_sn        - np.log(1e-3)) / 1.0)**2
    return nlml


# ─────────────────────────────────────────────
#  PIPELINE 2: LOCAL GPR CORRECTION
# ─────────────────────────────────────────────

def gpr_correction_2d(xy_dyn, u_dyn, mu_prior_dyn, var_prior_dyn,
                      xy_star, mu_prior_star, var_prior_star,
                      l_loc,ly, sigma_loc, sigma_n_loc,
                      dyn_x_left=0.0, dyn_x_right=1.0):
    """
    Fit a local GP to the residual u - mu_prior at dynamic points,
    then add the correction to the global posterior.

    Taper the correction smoothly in x so it vanishes outside the
    dynamic strip [dyn_x_left, dyn_x_right].
    """
    r_obs = u_dyn - mu_prior_dyn

    # Isotropic local RBF kernel (same lx=ly=l_loc)
    Kdd = k2d(xy_dyn, xy_dyn, l_loc, ly, sigma_loc)
    Kdd += (sigma_n_loc**2 + np.mean(var_prior_dyn)) * np.eye(len(xy_dyn))
    Kdd += 1e-8 * np.eye(len(xy_dyn))

    Ksd = k2d(xy_star, xy_dyn, l_loc, ly, sigma_loc)
    Kss_diag = sigma_loc**2 * np.ones(len(xy_star))

    L_fac = cho_factor(Kdd)
    alpha = cho_solve(L_fac, r_obs)

    delta_mu  = Ksd @ alpha
    V         = cho_solve(L_fac, Ksd.T)
    delta_var = Kss_diag - np.einsum('ij,ji->i', Ksd, V)
    delta_var = np.maximum(delta_var, 0.0)

    # Smooth taper in x only (boundary layer is a strip at x~1)
    x_star = xy_star[:, 0]
    width  = max(l_loc, 3 * (dyn_x_right - dyn_x_left) / np.sqrt(len(xy_dyn)))
    taper  = (1.0 / (1.0 + np.exp(-(x_star - dyn_x_left  + width) / (width/4)))
            * 1.0 / (1.0 + np.exp( (x_star - dyn_x_right - width) / (width/4))))

    delta_mu_tapered = delta_mu * taper
    mu_combined      = mu_prior_star + delta_mu_tapered
    var_combined     = var_prior_star + delta_var * taper

    return mu_combined, var_combined, delta_mu_tapered, delta_var


# ─────────────────────────────────────────────
#  DATA GENERATION
# ─────────────────────────────────────────────

# Interior observations (avoid the very thin boundary layer for Pipeline 1)
xy_obs_raw = np.column_stack([
    np.random.uniform(0.0, 1.0 - 10*DELTA, N_OBS),
    np.random.uniform(0.0, 1.0,            N_OBS)
])
u_obs = u_exact_2d(xy_obs_raw, KAPPA_TRUE, V_TRUE) \
      + np.random.normal(0, NOISE_OBS, N_OBS)

# Static interior collocation points (uniform grid, no endpoints)
nc_1d = int(np.ceil(np.sqrt(N_STATIC_C)))
_cx = np.linspace(0, 1, nc_1d + 2)[1:-1]
_cy = np.linspace(0, 1, nc_1d + 2)[1:-1]
_CX, _CY = np.meshgrid(_cx, _cy)
xy_col_static = np.column_stack([_CX.ravel(), _CY.ravel()])
f_col_static  = f_rhs_2d(xy_col_static, KAPPA_TRUE, V_TRUE) \
              + np.random.normal(0, NOISE_COL, len(xy_col_static))

# Boundary condition points: all four edges, u=0
n_per_edge = N_BC // 4
_e = np.linspace(0, 1, n_per_edge)
xy_bc = np.vstack([
    np.column_stack([_e,            np.zeros(n_per_edge)]),   # bottom y=0
    np.column_stack([_e,            np.ones(n_per_edge)]),    # top    y=1
    np.column_stack([np.zeros(n_per_edge), _e]),              # left   x=0
    np.column_stack([np.ones(n_per_edge),  _e]),              # right  x=1
])
u_bc = np.zeros(len(xy_bc))

# Dynamic points: dense strip near x=1 (outflow boundary layer)
xy_dyn = np.column_stack([
    np.random.uniform(DYN_X_LEFT, DYN_X_RIGHT, N_DYN),
    np.random.uniform(0.0,        1.0,          N_DYN)
])
u_dyn = u_exact_2d(xy_dyn, KAPPA_TRUE, V_TRUE) \
      + np.random.normal(0, NOISE_OBS, N_DYN)

# Prediction grid
_gx = np.linspace(0, 1, N_STAR_1D)
_gy = np.linspace(0, 1, N_STAR_1D)
_GX, _GY = np.meshgrid(_gx, _gy)
xy_star = np.column_stack([_GX.ravel(), _GY.ravel()])
u_true  = u_exact_2d(xy_star, KAPPA_TRUE, V_TRUE)

print("Problem setup (2D):")
print(f"  Pe = {Pe:.0f},  delta = {DELTA:.4f}")
print(f"  Boundary layer strip: x in [{DYN_X_LEFT:.4f}, {DYN_X_RIGHT:.1f}]")
print(f"  n_obs={N_OBS}, n_static_col={len(xy_col_static)}, "
      f"n_bc={len(xy_bc)}, n_dyn={N_DYN}")
print(f"  Prediction grid: {N_STAR_1D}x{N_STAR_1D} = {len(xy_star)} points")

# ─────────────────────────────────────────────
#  HYPERPARAMETER INITIAL VALUES AND BOUNDS
# ─────────────────────────────────────────────

THETA0_GPR = np.array([
    0.15,          # lx
    0.15,          # ly
    1.0,           # sigma
    np.log(1e-3),  # log sigma_n
])

BOUNDS_GPR = [
    (0.02, 0.5),   # lx
    (0.02, 0.5),   # ly
    (0.1,  5.0),   # sigma
    (-10,  -1),    # log sigma_n
]

# Local kernel for Pipeline 2
L_LOCAL    = max(3 * DELTA, 0.02)
SIGMA_LOCAL = 0.5
SN_LOCAL    = NOISE_OBS

# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("  Running 2D GPR pipeline (RBF kernel)")
print('='*60)

# ── Optimise hyperparameters ──
print("  Optimising hyperparameters...")
res = minimize(
    gpr_nlml_2d,
    THETA0_GPR,
    args=(xy_obs_raw, u_obs, xy_col_static, f_col_static, xy_bc, u_bc),
    method='L-BFGS-B',
    bounds=BOUNDS_GPR,
    options={'maxiter': 300, 'ftol': 1e-12}
)
lx_opt, ly_opt, sigma_opt, log_sn_opt = res.x
sn_opt = np.exp(log_sn_opt)
print(f"  lx={lx_opt:.4f}, ly={ly_opt:.4f}, sigma={sigma_opt:.4f}, sn={sn_opt:.2e}")

# ── Pipeline 1: global GPR ──
print("  Running global GPR...")
(mu1, var1, K_train, alpha_gpr) = gpr_posterior_2d(
    xy_obs_raw, u_obs, xy_col_static, f_col_static, xy_bc, u_bc,
    xy_star, lx_opt, ly_opt, sigma_opt, sn_opt, V_TRUE, KAPPA_TRUE)
std1 = np.sqrt(var1)
l2_1 = np.linalg.norm(mu1 - u_true) / np.linalg.norm(u_true)
print(f"  Global GPR L2 error: {l2_1:.4f}")

# ── GPR posterior at dynamic points (prior for Pipeline 2) ──
(mu1_dyn, var1_dyn, _, _) = gpr_posterior_2d(
    xy_obs_raw, u_obs, xy_col_static, f_col_static, xy_bc, u_bc,
    xy_dyn, lx_opt, ly_opt, sigma_opt, sn_opt, V_TRUE, KAPPA_TRUE)

# ── Pipeline 2: local GPR correction ──
print("  Running GPR correction...")
(mu2, var2, delta_mu, delta_var) = gpr_correction_2d(
    xy_dyn, u_dyn, mu1_dyn, var1_dyn,
    xy_star, mu1, var1,
    L_LOCAL,ly_opt ,SIGMA_LOCAL, SN_LOCAL,
    dyn_x_left=DYN_X_LEFT, dyn_x_right=DYN_X_RIGHT)
std2 = np.sqrt(np.maximum(var2, 0))
l2_2 = np.linalg.norm(mu2 - u_true) / np.linalg.norm(u_true)
print(f"  Combined L2 error:   {l2_2:.4f}")


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

SZ = N_STAR_1D   # grid size for reshape

COLORS = {
    'exact':    '#1a1a2e',
    'gpr':      '#e94560',
    'combined': '#0f3460',
    'dyn_pts':  '#f5a623',
    'obs_pts':  '#7ed321',
}

# Reshape flat arrays onto 2D grids for pcolormesh
def to_grid(arr):
    return arr.reshape(SZ, SZ)

GX = _GX   # (SZ, SZ)
GY = _GY

u_true_g  = to_grid(u_true)
mu1_g     = to_grid(mu1)
mu2_g     = to_grid(mu2)
std1_g    = to_grid(std1)
std2_g    = to_grid(std2)
err1_g    = to_grid(np.abs(mu1 - u_true))
err2_g    = to_grid(np.abs(mu2 - u_true))
delta_mu_g= to_grid(delta_mu)

fig = plt.figure(figsize=(26, 28))
fig.patch.set_facecolor('#f8f9fa')

nrows, ncols = 4, 3

def add_ax(pos, title, data, cmap='RdYlBu_r', vmin=None, vmax=None,
           scatter_xy=None, scatter_c='w', scatter_m='o', scatter_s=18,
           scatter_xy2=None, scatter_c2='orange', scatter_m2='^', scatter_s2=25,
           cb_label=''):
    ax = fig.add_subplot(nrows, ncols, pos)
    ax.set_facecolor('white')
    vm = np.nanmax(np.abs(data)) if vmin is None else None
    if vmin is None and vmax is None:
        vmin, vmax = -vm, vm
    pcm = ax.pcolormesh(GX, GY, data, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto', rasterized=True)
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=cb_label)
    # boundary layer strip
    ax.axvspan(DYN_X_LEFT, DYN_X_RIGHT, alpha=0.10, color='orange')
    if scatter_xy is not None:
        ax.scatter(scatter_xy[:, 0], scatter_xy[:, 1],
                   s=scatter_s, c=scatter_c, marker=scatter_m,
                   edgecolors='k', linewidths=0.3, zorder=5)
    if scatter_xy2 is not None:
        ax.scatter(scatter_xy2[:, 0], scatter_xy2[:, 1],
                   s=scatter_s2, c=scatter_c2, marker=scatter_m2,
                   edgecolors='k', linewidths=0.3, zorder=6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax

vrange = np.nanmax(np.abs(u_true_g))

# Row 1: Exact | Pipeline-1 mean | Pipeline-2 mean
add_ax(1,  'Exact solution u(x,y)',
       u_true_g, cmap='RdYlBu_r', vmin=-vrange, vmax=vrange,
       scatter_xy=xy_obs_raw, scatter_c='lime', scatter_m='o',
       cb_label='u')
add_ax(2,  'Pipeline 1 — Global GPR posterior mean',
       mu1_g, cmap='RdYlBu_r', vmin=-vrange, vmax=vrange,
       scatter_xy=xy_obs_raw, scatter_c='lime', scatter_m='o',
       cb_label='u')
add_ax(3,  'Pipeline 2 — Combined posterior mean',
       mu2_g, cmap='RdYlBu_r', vmin=-vrange, vmax=vrange,
       scatter_xy=xy_dyn, scatter_c='orange', scatter_m='^',
       cb_label='u')

# Row 2: GPR std | Combined std | Correction Δμ
add_ax(4,  'Pipeline 1 — Posterior std σ₁',
       std1_g, cmap='Oranges', vmin=0, vmax=None,
       cb_label='σ')
add_ax(5,  'Pipeline 2 — Posterior std σ₂',
       std2_g, cmap='Oranges', vmin=0, vmax=None,
       cb_label='σ')
add_ax(6,  'GPR correction Δμ = μ₂ − μ₁',
       delta_mu_g, cmap='PuOr',
       scatter_xy=xy_dyn, scatter_c='orange', scatter_m='^',
       cb_label='Δμ')

# Row 3: Absolute errors
emax = max(np.nanmax(err1_g), np.nanmax(err2_g))
add_ax(7,  f'Pipeline 1 — |error|   (L2={l2_1:.3f})',
       err1_g, cmap='hot_r', vmin=0, vmax=emax, cb_label='|u−û|')
add_ax(8,  f'Pipeline 2 — |error|   (L2={l2_2:.3f})',
       err2_g, cmap='hot_r', vmin=0, vmax=emax, cb_label='|u−û|')

# Row 3, col 3: log-error ratio
log_ratio = np.log10(np.maximum(err2_g, 1e-12) / np.maximum(err1_g, 1e-12))
ax_lr = add_ax(9, 'log₁₀(|err₂|/|err₁|)  (<0 = correction helps)',
               log_ratio, cmap='coolwarm_r', cb_label='log ratio')

# Row 4: 1D slices at y=0.5 (mid-plane), full domain and boundary-layer zoom
y_mid_idx = SZ // 2
x_plot    = _gx

def _slice(arr2d): return arr2d[y_mid_idx, :]

ax_full = fig.add_subplot(nrows, ncols, 10)
ax_full.set_facecolor('white')
ax_full.plot(x_plot, _slice(u_true_g),  '--', color=COLORS['exact'],    lw=2.5, label='Exact')
ax_full.plot(x_plot, _slice(mu1_g),      '-', color=COLORS['gpr'],      lw=2,   label='GPR')
ax_full.plot(x_plot, _slice(mu2_g),      '-', color=COLORS['combined'], lw=2,   label='Combined')
ax_full.fill_between(x_plot,
    _slice(mu1_g) - 2*_slice(std1_g),
    _slice(mu1_g) + 2*_slice(std1_g),
    alpha=0.15, color=COLORS['gpr'])
ax_full.fill_between(x_plot,
    _slice(mu2_g) - 2*_slice(std2_g),
    _slice(mu2_g) + 2*_slice(std2_g),
    alpha=0.15, color=COLORS['combined'])
ax_full.axvspan(DYN_X_LEFT, DYN_X_RIGHT, alpha=0.1, color='orange', label='Dynamic region')
ax_full.set_title('Mid-plane slice  y=0.5  (full domain)', fontsize=10, fontweight='bold')
ax_full.set_xlabel('x'); ax_full.set_ylabel('u(x, 0.5)')
ax_full.legend(fontsize=8); ax_full.grid(True, alpha=0.3)

ax_zoom = fig.add_subplot(nrows, ncols, 11)
ax_zoom.set_facecolor('white')
zm = x_plot >= DYN_X_LEFT - 0.05
ax_zoom.plot(x_plot[zm], _slice(u_true_g)[zm],  '--', color=COLORS['exact'],    lw=2.5, label='Exact')
ax_zoom.plot(x_plot[zm], _slice(mu1_g)[zm],      '-', color=COLORS['gpr'],      lw=2,   label='GPR')
ax_zoom.plot(x_plot[zm], _slice(mu2_g)[zm],      '-', color=COLORS['combined'], lw=2,   label='Combined')
ax_zoom.fill_between(x_plot[zm],
    (_slice(mu1_g) - 2*_slice(std1_g))[zm],
    (_slice(mu1_g) + 2*_slice(std1_g))[zm],
    alpha=0.2, color=COLORS['gpr'])
ax_zoom.fill_between(x_plot[zm],
    (_slice(mu2_g) - 2*_slice(std2_g))[zm],
    (_slice(mu2_g) + 2*_slice(std2_g))[zm],
    alpha=0.2, color=COLORS['combined'])
ax_zoom.axvspan(DYN_X_LEFT, DYN_X_RIGHT, alpha=0.1, color='orange')
ax_zoom.set_title('Mid-plane slice  y=0.5  (boundary layer zoom)', fontsize=10, fontweight='bold')
ax_zoom.set_xlabel('x'); ax_zoom.set_ylabel('u(x, 0.5)')
ax_zoom.set_xlim(DYN_X_LEFT - 0.05, 1.0)
ax_zoom.legend(fontsize=8); ax_zoom.grid(True, alpha=0.3)

# Row 4, col 3: point locations summary
ax_pts = fig.add_subplot(nrows, ncols, 12)
ax_pts.set_facecolor('white')
ax_pts.scatter(xy_obs_raw[:, 0],    xy_obs_raw[:, 1],    s=12, c='lime',   label=f'Observations ({N_OBS})',        zorder=4)
ax_pts.scatter(xy_col_static[:, 0], xy_col_static[:, 1], s=6,  c='purple', label=f'Static colloc ({len(xy_col_static)})', zorder=3, alpha=0.6, marker='s')
ax_pts.scatter(xy_bc[:, 0],         xy_bc[:, 1],         s=10, c='royalblue', label=f'BC points ({len(xy_bc)})',  zorder=5)
ax_pts.scatter(xy_dyn[:, 0],        xy_dyn[:, 1],        s=20, c='orange', label=f'Dynamic ({N_DYN})',            zorder=6, marker='^')
ax_pts.axvspan(DYN_X_LEFT, DYN_X_RIGHT, alpha=0.1, color='orange')
ax_pts.set_title('Training point layout', fontsize=10, fontweight='bold')
ax_pts.set_xlabel('x'); ax_pts.set_ylabel('y')
ax_pts.set_xlim(0, 1); ax_pts.set_ylim(0, 1)
ax_pts.legend(fontsize=7, loc='upper left')
ax_pts.grid(True, alpha=0.3)

plt.suptitle(
    f'2D Advection-Diffusion on [0,1]²  |  Pe = {Pe:.0f}  |  '
    f'v=({V_TRUE},0), κ={KAPPA_TRUE}  |  δ≈{DELTA:.4f}\n'
    f'RBF GPR  |  lx={lx_opt:.3f}, ly={ly_opt:.3f}, σ={sigma_opt:.3f}, σₙ={sn_opt:.1e}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.42, wspace=0.35)
plt.savefig('advdiff_2d_unified.png', dpi=150, bbox_inches='tight')
plt.show()

if PLOTLY_AVAILABLE:
    try:
        fig3d = go.Figure(
            data=[go.Surface(
                x=_GX,
                y=_GY,
                z=mu2_g,
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
    except Exception:
        pass
# End of script 

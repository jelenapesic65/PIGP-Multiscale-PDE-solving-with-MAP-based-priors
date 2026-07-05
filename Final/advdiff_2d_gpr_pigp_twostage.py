"""
Comparison of GPR and PIGP
2D steady advection-diffusion, [0,1]^2,Pe = 500

3 cases:
    'sinsin'   f = sin(pi x) sin(pi y)  -> closed-form exact solution
    'two_blob' two Gaussian sources     -> high-accuracy FD reference
    'complex'  11 mixed-sign sources    -> high-accuracy FD reference
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator as RGI
import warnings
warnings.filterwarnings('ignore')

CASE       = 'complex'      # 'sinsin' | 'two_blob' | 'complex'
SHOW_3D    = True           # show 3D surface plots of the prediction          

V_TRUE     = 5.0
KAPPA_TRUE = 0.01
Pe         = V_TRUE / KAPPA_TRUE
DELTA      = KAPPA_TRUE / V_TRUE          # boundary-layer thickness ~ 1/Pe

N_OBS      = 160     # scattered interior sensors 
NC_1D      = 20     # collocation points per axis (uniform)
N_BC       = 30     # boundary points (u = 0)
N_STAR_1D  = 60     # prediction grid per axis

NOISE_OBS  = 1e-3
NOISE_COL  = 1e-3

# --- two-stage (residual-correction) settings ---------------------------------
# stage 1 : PIGP WITHOUT boundary conditions  -> its mean is the stage-2 prior mean
# stage 2 : residual GP on focused collocation pts (placed by the PDE residual)
#           + boundary points; final solution = stage1 + stage2
N_BC2_EDGE     = 24      # stage-2 boundary points PER EDGE (u=0 is known exactly,
                         # so stage 2 may sample the boundary as densely as it likes)
N_FOCUS        = 60      # residual-driven focused collocation points (stage 2)
N_ANCHOR       = 24      # low-residual "anchor" collocation pts (pin delta->0 in bulk)
FOCUS_MIN_DIST = 0.012   # min spacing of focused pts  (Gram-conditioning guard)
ANCHOR_MIN_DIST= 0.08    # min spacing of anchors / around focused pts
CAND_NX        = 160     # residual candidate grid, x (graded toward outflow)
CAND_NY        = 70      # residual candidate grid, y (uniform)
# stage-2 lengthscale = (scale) x (inherited stage-1 anisotropic lengthscale).
#   1.0, 1.0 = literal inherit -- but then the boundary-value correction
#   (delta = -mu1|_bc) spreads ~lx into the interior and pollutes the bulk that
#   stage 1 already fit. The correction should be SHORT across the layer (x) and
#   LONG along it (y), so a strongly anisotropic scaling localises it. Set both
#   to 1.0 to recover the pure-inherit kernel.
LX_S2, LY_S2   = 0.07, 1.0

np.random.seed(42)

#forcings
def f_sinsin(xy):
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

_BLOBS_2 = [(0.30, 0.30, 0.09, 0.09, 6.0),
            (0.70, 0.65, 0.07, 0.07, 4.5)]

_BLOBS_C = [(0.20, 0.30, 0.05, 0.07,  6.0), (0.35, 0.68, 0.04, 0.04, -5.0),
            (0.52, 0.42, 0.06, 0.05,  5.5), (0.50, 0.80, 0.05, 0.05, -4.0),
            (0.66, 0.25, 0.04, 0.06,  6.5), (0.70, 0.60, 0.05, 0.04, -6.0),
            (0.80, 0.45, 0.035,0.035, 5.0), (0.30, 0.15, 0.04, 0.04,  4.5),
            (0.82, 0.78, 0.045,0.05, -4.5), (0.15, 0.55, 0.05, 0.045, 4.0),
            (0.45, 0.55, 0.03, 0.03, -4.0)]

def _sum_blobs(xy, blobs):
    x, y = xy[:, 0], xy[:, 1]
    out = np.zeros_like(x)
    for cx, cy, sx, sy, a in blobs:
        out += a * np.exp(-((x-cx)**2/(2*sx**2) + (y-cy)**2/(2*sy**2)))
    return out

def f_two_blob(xy):
    return _sum_blobs(xy, _BLOBS_2)

def f_complex(xy):
    x, y = xy[:, 0], xy[:, 1]
    tex = 1.2*np.sin(6*np.pi*x)*np.sin(5*np.pi*y)*np.exp(-((x-.5)**2+(y-.5)**2)/(2*.35**2))
    return _sum_blobs(xy, _BLOBS_C) + tex

FORCING = {'sinsin': f_sinsin, 'two_blob': f_two_blob, 'complex': f_complex}[CASE]

#reference solutions
def u1d_exact(x, kappa, v):

    pi = np.pi
    denom = pi**2 * kappa**2 + v**2
    A =  kappa / denom
    B = -v / (pi * denom)
    r = v / kappa
    em = np.exp(-r)                        
    C2 = 2*B*em / (1 - em)
    C1 = -B - C2
    bl = 2*B*np.exp(r*(x - 1.0)) / (1 - em)  
    return C1 + bl + A*np.sin(pi*x) + B*np.cos(pi*x)

def u_exact_sinsin(xy, kappa, v):
    return u1d_exact(xy[:, 0], kappa, v) * np.sin(np.pi * xy[:, 1])

def _graded_x(Nx, beta):
    xi = np.linspace(0.0, 1.0, Nx)
    x = 1.0 - np.sinh(beta*(1.0 - xi)) / np.sinh(beta)
    x[0], x[-1] = 0.0, 1.0
    return x

def solve_reference_fd(forcing, v, kappa, Nx=900, Ny=240, beta=3.2):

    x = _graded_x(Nx, beta); y = np.linspace(0.0, 1.0, Ny)
    hy = y[1] - y[0]
    hm = x[1:] - x[:-1]; hp, hmm = hm[1:], hm[:-1]; s = hp + hmm
    d1m, d10, d1p = -hp/(hmm*s), (hp-hmm)/(hp*hmm), hmm/(hp*s)
    d2m, d20, d2p = 2/(hmm*s), -2/(hp*hmm), 2/(hp*s)
    rows, cols, vals = [], [], []
    def add(k, kk, val): rows.append(k); cols.append(kk); vals.append(val)
    for i in range(Nx):
        for j in range(Ny):
            k = i*Ny + j
            if i in (0, Nx-1) or j in (0, Ny-1):
                add(k, k, 1.0); continue
            ii = i-1
            add(k, (i-1)*Ny+j, v*d1m[ii]-kappa*d2m[ii])
            add(k, k,          v*d10[ii]-kappa*d20[ii] + 2*kappa/hy**2)
            add(k, (i+1)*Ny+j, v*d1p[ii]-kappa*d2p[ii])
            add(k, i*Ny+(j-1), -kappa/hy**2)
            add(k, i*Ny+(j+1), -kappa/hy**2)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx*Ny, Nx*Ny))
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    b = forcing(np.c_[Xg.ravel(), Yg.ravel()])
    bmask = np.zeros((Nx, Ny), bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    b[bmask.ravel()] = 0.0
    u = spla.spsolve(A.tocsc(), b).reshape(Nx, Ny)
    cellPe = (v*np.diff(x)/(2*kappa)).max()
    print(f"  [FD reference] Nx={Nx} Ny={Ny}  cellPe_max={cellPe:.3f}  "
          f"u in [{u.min():.3f}, {u.max():.3f}]")
    return RGI((x, y), u, bounds_error=False, fill_value=0.0)

if CASE == 'sinsin':
    ref = lambda xy: u_exact_sinsin(xy, KAPPA_TRUE, V_TRUE)
else:
    _interp = solve_reference_fd(FORCING, V_TRUE, KAPPA_TRUE)
    ref = lambda xy: _interp(xy)

# kernels (closed form)
def _rbf_factors(p, q, lx, ly, sigma):
    rx = p[:, 0][:, None] - q[:, 0][None, :]
    ry = p[:, 1][:, None] - q[:, 1][None, :]
    lx2, ly2 = lx**2, ly**2
    K = sigma**2 * np.exp(-0.5*rx**2/lx2) * np.exp(-0.5*ry**2/ly2)
    return K, rx, ry, rx/lx2, ry/ly2, lx2, ly2

def k2d(p, q, lx, ly, sigma):
    return _rbf_factors(p, q, lx, ly, sigma)[0]

def k2d_L2(p, q, lx, ly, sigma, v, kappa):
    K, rx, ry, ex, ey, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)
    Lq = v*ex - kappa*(ex**2 - 1.0/lx2) - kappa*(ey**2 - 1.0/ly2)
    return Lq * K

def k2d_L1(p, q, lx, ly, sigma, v, kappa):
    # operator applied to the FIRST argument p:  L_p k(p,q) = Cov(Lu(p), u(q)).
    # d/dp_x r = +1 (vs -1 for q) -> advection term flips sign; the (symmetric)
    # diffusion term is unchanged. Needed to evaluate  L(mu1)  analytically.
    K, rx, ry, ex, ey, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)
    Lp = -v*ex - kappa*(ex**2 - 1.0/lx2) - kappa*(ey**2 - 1.0/ly2)
    return Lp * K

def k2d_L1L2(p, q, lx, ly, sigma, v, kappa):
    K, rx, ry, ex, ey, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)
    Lq        = v*ex - kappa*(ex**2 - 1.0/lx2) - kappa*(ey**2 - 1.0/ly2)
    dLq_dx1   = (v - 2*kappa*ex) / lx2
    d2Lq_dx12 = -2*kappa / lx2**2
    dLq_dy1   = -2*kappa*ey / ly2
    d2Lq_dy12 = -2*kappa / ly2**2
    dF_dx1    = dLq_dx1 - ex*Lq 
    d2F_dx12  = d2Lq_dx12 - Lq/lx2 - 2*ex*dLq_dx1 + ex**2*Lq
    d2F_dy12  = d2Lq_dy12 - Lq/ly2 - 2*ey*dLq_dy1 + ey**2*Lq
    return (v*dF_dx1 - kappa*(d2F_dx12 + d2F_dy12)) * K

#GP system
def build_system(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                 lx, ly, sigma, sigma_n, v, kappa, use_physics):
    #use_physics=False - plain GPR: no knowledge of boundary conditions,
    #trained on interior observations only.

    #Per-block noise: BC ~ exact, obs ~ sigma_n (data noise, a hyperparameter),
    #collocation ~ NOISE_COL (fixed PDE-residual tolerance).  A single shared
    #noise fails here because the value targets (~O(u)) and the operator
   # targets f (~O(v*u/l)) live on very different scales."""
    if use_physics:
        xy_val = np.vstack([xy_bc, xy_obs])
        u_val  = np.concatenate([u_bc, u_obs])
        nz_v = np.concatenate([np.full(len(xy_bc), 1e-8),
                               np.full(len(xy_obs), sigma_n**2)])
    
        #xy_val = xy_obs
        #u_val  = u_obs
        #nz_v = np.full(len(xy_obs), sigma_n**2)
    else:
        xy_val = xy_obs
        u_val  = u_obs
        nz_v = np.full(len(xy_obs), sigma_n**2)
    n_v = len(xy_val)
    K_vv = k2d(xy_val, xy_val, lx, ly, sigma)
    if use_physics and len(xy_col):
        n_c = len(xy_col); n = n_v + n_c
        K_vc = k2d_L2(xy_val, xy_col, lx, ly, sigma, v, kappa)
        K_cc = k2d_L1L2(xy_col, xy_col, lx, ly, sigma, v, kappa)
        K = np.zeros((n, n))
        K[:n_v, :n_v] = K_vv; K[:n_v, n_v:] = K_vc
        K[n_v:, :n_v] = K_vc.T; K[n_v:, n_v:] = K_cc
        y = np.concatenate([u_val, f_col])
        nz = np.concatenate([nz_v, np.full(n_c, NOISE_COL**2)])
    else:
        K = K_vv.copy(); y = u_val; nz = nz_v
    K += np.diag(nz) + 1e-8*np.eye(len(K))
    return xy_val, y, K, n_v

def gp_posterior(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc, xy_star,
                 lx, ly, sigma, sigma_n, v, kappa, use_physics):
    xy_val, y, K, n_v = build_system(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                                     lx, ly, sigma, sigma_n, v, kappa, use_physics)
    Lf = cho_factor(K); alpha = cho_solve(Lf, y)
    Ks = k2d(xy_star, xy_val, lx, ly, sigma)
    if use_physics and len(xy_col):
        Ks = np.hstack([Ks, k2d_L2(xy_star, xy_col, lx, ly, sigma, v, kappa)])
    mu = Ks @ alpha
    var = sigma**2 - np.einsum('ij,ji->i', Ks, cho_solve(Lf, Ks.T))
    return mu, np.maximum(var, 0.0)

def nlml(theta, xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc, v, kappa, use_physics, sn):
    
    lx, ly, sigma = theta
    _, y, K, _ = build_system(xy_obs, u_obs, xy_col, f_col, xy_bc, u_bc,
                              lx, ly, sigma, sn, v, kappa, use_physics)
    sign, ld = slogdet(K)
    if sign <= 0: return 1e10
    val = 0.5*(y @ solve(K, y)) + 0.5*ld
    val -= -0.5*((np.log(lx)-np.log(0.15))/2.0)**2
    val -= -0.5*((np.log(ly)-np.log(0.15))/2.0)**2
    val -= -0.5*((np.log(sigma)-np.log(1.0))/1)**2
    return val

# data
xy_obs = np.random.uniform(0.0, 1.0, size=(N_OBS, 2))
u_obs  = ref(xy_obs) + np.random.normal(0, NOISE_OBS, N_OBS)

_c = np.linspace(0, 1, NC_1D + 2)[1:-1]
_CX, _CY = np.meshgrid(_c, _c)
xy_col = np.column_stack([_CX.ravel(), _CY.ravel()])
f_col  = FORCING(xy_col) + np.random.normal(0, NOISE_COL, len(xy_col))

npe = N_BC // 4; _e = np.linspace(0, 1, npe)
xy_bc = np.vstack([np.c_[_e, np.zeros(npe)], np.c_[_e, np.ones(npe)],
                   np.c_[np.zeros(npe), _e], np.c_[np.ones(npe), _e]])
u_bc = np.zeros(len(xy_bc))

_g = np.linspace(0, 1, N_STAR_1D)
_GX, _GY = np.meshgrid(_g, _g)
xy_star = np.column_stack([_GX.ravel(), _GY.ravel()])
u_true = ref(xy_star)

print(f"Case '{CASE}':  Pe={Pe:.0f}, delta={DELTA:.4f}")
print(f"  n_obs={N_OBS}, n_col={len(xy_col)} (uniform {NC_1D}x{NC_1D}), n_bc={len(xy_bc)}")

# --- standardise targets to unit scale (L is linear: u/S solves L(u/S)=f/S) ---
S = np.std(u_obs)
u_obs_n, u_bc_n, f_col_n = u_obs/S, u_bc/S, f_col/S
SN_FIX = NOISE_OBS / S                     # known sensor noise, standardised

#fit both models
def fit(use_physics, theta0):
    r = minimize(nlml, theta0,
                 args=(xy_obs, u_obs_n, xy_col, f_col_n, xy_bc, u_bc_n,
                       V_TRUE, KAPPA_TRUE, use_physics, SN_FIX),
                 method='Nelder-Mead',
                 options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 600})
    lx, ly, sg = r.x
    mu, var = gp_posterior(xy_obs, u_obs_n, xy_col, f_col_n, xy_bc, u_bc_n, xy_star,
                           lx, ly, sg, SN_FIX, V_TRUE, KAPPA_TRUE, use_physics)
    return dict(lx=lx, ly=ly, sigma=sg, sn=SN_FIX, mu=mu*S, std=np.sqrt(var)*S)

# ======================================================================
#  TWO-STAGE  (residual correction)
#  stage 1 : PIGP WITHOUT BCs  ->  posterior mean mu1 is the prior mean of stage 2
#  stage 2 : GP for the correction  delta = u - mu1, conditioned on
#              * focused collocation pts (op. target  f - L(mu1), i.e. residual)
#              * boundary pts           (value target  0 - mu1|_bc = -mu1|_bc)
#            mixed kernel:  value-value = k,  val-col = k_L2,  col-col = k_L1L2
#  final   : u = mu1 + delta
#  Everything below runs in the STANDARDISED scale (u/S); *S at the very end.
# ======================================================================
_EMPTY_XY = np.empty((0, 2)); _EMPTY_U = np.empty((0,))

def fit_stage1(theta0):
    # PIGP with obs + uniform collocation, but NO boundary points.
    r = minimize(nlml, theta0,
                 args=(xy_obs, u_obs_n, xy_col, f_col_n, _EMPTY_XY, _EMPTY_U,
                       V_TRUE, KAPPA_TRUE, True, SN_FIX),
                 method='Nelder-Mead',
                 options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 600})
    lx, ly, sg = r.x
    xy_val, y, K, n_v = build_system(xy_obs, u_obs_n, xy_col, f_col_n,
                                     _EMPTY_XY, _EMPTY_U, lx, ly, sg,
                                     SN_FIX, V_TRUE, KAPPA_TRUE, True)
    alpha = cho_solve(cho_factor(K), y)
    return dict(lx=lx, ly=ly, sigma=sg, alpha=alpha,
                xy_val=xy_val, xy_col=xy_col, n_v=n_v)

def stage1_eval(m1, xq):
    # returns (mu1, L mu1) at xq, analytically, in standardised units.
    lx, ly, sg = m1['lx'], m1['ly'], m1['sigma']; v, kap = V_TRUE, KAPPA_TRUE
    nv = m1['n_v']; av, ac = m1['alpha'][:nv], m1['alpha'][nv:]
    Xv, Xc = m1['xy_val'], m1['xy_col']
    mu  = k2d(xq, Xv, lx, ly, sg) @ av \
        + k2d_L2(xq, Xc, lx, ly, sg, v, kap) @ ac
    Lmu = k2d_L1(xq, Xv, lx, ly, sg, v, kap) @ av \
        + k2d_L1L2(xq, Xc, lx, ly, sg, v, kap) @ ac
    return mu, Lmu

def _greedy_pick(cand, order, min_dist, n_max, seed_pts=None):
    # walk candidates in the given priority order, accept if far enough from
    # everything accepted so far (and from seed_pts). Keeps the Gram matrix
    # of the double-operator block away from singularity.
    picked, pts = [], ([] if seed_pts is None else [p for p in seed_pts])
    for idx in order:
        p = cand[idx]
        if not pts or np.min(np.linalg.norm(np.asarray(pts) - p, axis=1)) >= min_dist:
            picked.append(idx); pts.append(p)
        if len(picked) >= n_max: break
    return picked

def place_focused_points(m1):
    # candidate grid graded toward the outflow (where the layer / residual lives)
    xcg = _graded_x(CAND_NX, 3.5); ycg = np.linspace(0.0, 1.0, CAND_NY)
    Xc, Yc = np.meshgrid(xcg, ycg, indexing='ij')
    cand = np.column_stack([Xc.ravel(), Yc.ravel()])
    keep = ((cand[:, 0] > 1e-6) & (cand[:, 0] < 1 - 1e-6) &
            (cand[:, 1] > 1e-6) & (cand[:, 1] < 1 - 1e-6))
    cand = cand[keep]
    _, Lmu1 = stage1_eval(m1, cand)
    res = np.abs(FORCING(cand) / S - Lmu1)          # |f - L mu1|  (standardised)
    hi  = np.argsort(-res)                            # high-residual first
    xy_focus  = cand[_greedy_pick(cand, hi, FOCUS_MIN_DIST, N_FOCUS)]
    lo  = np.argsort(res)                             # low-residual anchors
    xy_anchor = cand[_greedy_pick(cand, lo, ANCHOR_MIN_DIST, N_ANCHOR,
                                  seed_pts=list(xy_focus))]
    return xy_focus, xy_anchor, cand, res

def _dense_boundary(npe):
    e = np.linspace(0.0, 1.0, npe)
    return np.vstack([np.c_[e, np.zeros(npe)], np.c_[e, np.ones(npe)],
                      np.c_[np.zeros(npe), e], np.c_[np.ones(npe), e]])

def stage2_solve(m1, xy_col2, xy_bc2, xy_star_):
    lx, ly, sg = m1['lx'] * LX_S2, m1['ly'] * LY_S2, m1['sigma']  # inherited kernel
    v, kap = V_TRUE, KAPPA_TRUE
    # value block = boundary points:  target delta = u_bc - mu1(bc) = -mu1(bc)  (u_bc=0)
    mu1_bc, _ = stage1_eval(m1, xy_bc2)
    y_val = -mu1_bc
    # collocation block = focused (+anchor) points: target = f - L mu1  (residual)
    _, Lmu1_c = stage1_eval(m1, xy_col2)
    f_c   = (FORCING(xy_col2) + np.random.normal(0, NOISE_COL, len(xy_col2))) / S
    y_col = f_c - Lmu1_c
    nv, nc = len(xy_bc2), len(xy_col2); n = nv + nc
    Kg = np.zeros((n, n))
    Kg[:nv, :nv] = k2d(xy_bc2, xy_bc2, lx, ly, sg)                     # val - val
    Kg[:nv, nv:] = k2d_L2(xy_bc2, xy_col2, lx, ly, sg, v, kap)         # val - col (mixed)
    Kg[nv:, :nv] = Kg[:nv, nv:].T
    Kg[nv:, nv:] = k2d_L1L2(xy_col2, xy_col2, lx, ly, sg, v, kap)      # col - col (double op)
    nz = np.concatenate([np.full(nv, 1e-8), np.full(nc, NOISE_COL**2)])
    Kg += np.diag(nz) + 1e-8 * np.eye(n)
    a2 = cho_solve(cho_factor(Kg), np.concatenate([y_val, y_col]))
    # predict the correction at the grid: value block (k) + operator block (k_L2)
    Ks = np.hstack([k2d(xy_star_, xy_bc2, lx, ly, sg),
                    k2d_L2(xy_star_, xy_col2, lx, ly, sg, v, kap)])
    return Ks @ a2

def fit_two_stage(theta0=np.array([0.3, 0.3, 1.0])):
    m1 = fit_stage1(theta0)
    xy_focus, xy_anchor, cand, res = place_focused_points(m1)
    xy_col2 = np.vstack([xy_focus, xy_anchor])
    xy_bc2  = _dense_boundary(N_BC2_EDGE)         # u=0 known exactly -> sample densely
    delta_n = stage2_solve(m1, xy_col2, xy_bc2, xy_star)
    mu1_star, _ = stage1_eval(m1, xy_star)
    return dict(lx=m1['lx'], ly=m1['ly'], sigma=m1['sigma'], sn=SN_FIX,
                mu=(mu1_star + delta_n) * S,          # final = stage1 + stage2
                mu_s1=mu1_star * S,                   # stage-1 only (prior mean)
                mu_delta=delta_n * S,                 # stage-2 correction
                xy_focus=xy_focus, xy_anchor=xy_anchor, xy_bc2=xy_bc2,
                cand=cand, res=res)

print("  fitting plain GPR ...")
GPR  = fit(False, np.array([0.15, 0.15, 1.0]))
print("  fitting PIGP ...")
PIGP = fit(True,  np.array([0.3, 0.3, 1.0]))
print("  fitting two-stage (stage1 = PIGP w/o BCs -> prior mean of stage2) ...")
TWO  = fit_two_stage(np.array([0.3, 0.3, 1.0]))

def relL2(mu, m=None):
    if m is None: return np.linalg.norm(mu-u_true)/np.linalg.norm(u_true)
    return np.linalg.norm((mu-u_true)[m])/np.linalg.norm(u_true[m])
def rmse(mu, m):   # absolute; used for the near-outflow band where u->0
    return np.sqrt(np.mean((mu-u_true)[m]**2))

bulk = xy_star[:, 0] <= 0.96              # relative L2 well-defined here
layer = xy_star[:, 0] >= 0.90             # near-outflow band (u->0: use RMSE)
for name, M in [('plain GPR', GPR), ('PIGP', PIGP), ('two-stage', TWO)]:
    print(f"  {name:10s}: lx={M['lx']:.3f} ly={M['ly']:.3f} sigma={M['sigma']:.3f} sn={M.get('sn',SN_FIX):.1e}")
print(f"  two-stage focused pts: {len(TWO['xy_focus'])}  anchors: {len(TWO['xy_anchor'])}")
print("\n=============== error ===============")
print("                  full relL2   bulk relL2   outflow RMSE(x>=.90)")
print(f"plain GPR         {relL2(GPR['mu']):.4f}      {relL2(GPR['mu'],bulk):.4f}       {rmse(GPR['mu'],layer):.4e}")
print(f"PIGP              {relL2(PIGP['mu']):.4f}      {relL2(PIGP['mu'],bulk):.4f}       {rmse(PIGP['mu'],layer):.4e}")
print(f"two-stage S1 only {relL2(TWO['mu_s1']):.4f}      {relL2(TWO['mu_s1'],bulk):.4f}       {rmse(TWO['mu_s1'],layer):.4e}")
print(f"two-stage S1+S2   {relL2(TWO['mu']):.4f}      {relL2(TWO['mu'],bulk):.4f}       {rmse(TWO['mu'],layer):.4e}")
print(f"improvement       {relL2(GPR['mu'])/relL2(PIGP['mu']):.1f}x        "
      f"{relL2(GPR['mu'],bulk)/relL2(PIGP['mu'],bulk):.1f}x         "
      f"{rmse(GPR['mu'],layer)/rmse(PIGP['mu'],layer):.1f}x")
print(f"S1->S1+S2 gain    {relL2(TWO['mu_s1'])/relL2(TWO['mu']):.2f}x        "
      f"{relL2(TWO['mu_s1'],bulk)/relL2(TWO['mu'],bulk):.2f}x         "
      f"{rmse(TWO['mu_s1'],layer)/rmse(TWO['mu'],layer):.2f}x")

#figure
SZ = N_STAR_1D
g = lambda a: a.reshape(SZ, SZ)
u_g, gpr_g, pi_g = g(u_true), g(GPR['mu']), g(PIGP['mu'])
two_g, s1_g, dl_g = g(TWO['mu']), g(TWO['mu_s1']), g(TWO['mu_delta'])
etwo_g = g(np.abs(TWO['mu']-u_true))
egpr_g, epi_g = g(np.abs(GPR['mu']-u_true)), g(np.abs(PIGP['mu']-u_true))
l2_gpr, l2_pi, l2_two = relL2(GPR['mu']), relL2(PIGP['mu']), relL2(TWO['mu'])
VR = np.nanmax(np.abs(u_g)); EMAX = max(egpr_g.max(), epi_g.max())
BL_LEFT = max(0.0, 1.0 - 15*DELTA)

fig, ax = plt.subplots(3, 3, figsize=(17, 14)); fig.patch.set_facecolor('#f8f9fa')
def pc(a, d, t, cmap='RdYlBu_r', vmin=None, vmax=None, cb=''):
    a.set_facecolor('white')
    if vmin is None: vm = np.nanmax(np.abs(d)); vmin, vmax = -vm, vm
    m = a.pcolormesh(_GX, _GY, d, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    fig.colorbar(m, ax=a, fraction=0.046, pad=0.04, label=cb)
    a.axvspan(BL_LEFT, 1.0, alpha=0.08, color='orange')
    a.set_title(t, fontsize=10, fontweight='bold'); a.set_xlabel('x'); a.set_ylabel('y')
    a.set_xlim(0, 1); a.set_ylim(0, 1)

pc(ax[0,0], u_g, f'Exact / FD reference  u(x,y)   [{CASE}]', vmin=-VR, vmax=VR, cb='u')
ax[0,0].scatter(xy_obs[:,0], xy_obs[:,1], s=10, c='lime', edgecolors='k', lw=0.3, zorder=5, label='obs')
ax[0,0].legend(fontsize=7, loc='upper left')
pc(ax[0,1], gpr_g, f'plain GPR mean  (L2={l2_gpr:.3f})', vmin=-VR, vmax=VR, cb='u')
pc(ax[0,2], pi_g,  f'PIGP mean  (L2={l2_pi:.3f})',       vmin=-VR, vmax=VR, cb='u')

pc(ax[1,0], egpr_g, 'plain GPR |error|', cmap='hot_r', vmin=0, vmax=EMAX, cb='|u-û|')
pc(ax[1,1], epi_g,  'PIGP |error|',      cmap='hot_r', vmin=0, vmax=EMAX, cb='|u-û|')
ax[1,2].set_facecolor('white')
ax[1,2].scatter(xy_obs[:,0], xy_obs[:,1], s=18, c='lime', edgecolors='k', lw=.3, label=f'obs ({N_OBS})')
ax[1,2].scatter(xy_col[:,0], xy_col[:,1], s=12, c='purple', marker='s', alpha=.5, label=f'colloc ({len(xy_col)})')
ax[1,2].scatter(xy_bc[:,0], xy_bc[:,1], s=18, c='royalblue', edgecolors='k', lw=.3, label=f'BC ({len(xy_bc)})')
ax[1,2].scatter(TWO['xy_focus'][:,0], TWO['xy_focus'][:,1], s=22, c='red', marker='*',
                edgecolors='k', lw=.3, zorder=6, label=f"S2 focused ({len(TWO['xy_focus'])})")
ax[1,2].scatter(TWO['xy_anchor'][:,0], TWO['xy_anchor'][:,1], s=14, c='gold', marker='D',
                edgecolors='k', lw=.3, zorder=6, label=f"S2 anchors ({len(TWO['xy_anchor'])})")
ax[1,2].axvspan(BL_LEFT, 1.0, alpha=.08, color='orange')
ax[1,2].set_title('Training point layout', fontsize=10, fontweight='bold')
ax[1,2].set_xlim(0,1); ax[1,2].set_ylim(0,1); ax[1,2].legend(fontsize=7, loc='upper left'); ax[1,2].grid(alpha=.3)

ym = SZ//2
ax[2,0].set_facecolor('white')
ax[2,0].plot(_g, u_g[ym], '--', c='#1a1a2e', lw=2.5, label='reference')
ax[2,0].plot(_g, gpr_g[ym], '-', c='#3b6fb0', lw=2, label='plain GPR')
ax[2,0].plot(_g, pi_g[ym], '-', c='#e94560', lw=2, label='PIGP')
ax[2,0].plot(_g, two_g[ym], '-', c='#2a9d3f', lw=2, label='two-stage')
ax[2,0].axvspan(BL_LEFT, 1.0, alpha=.08, color='orange')
ax[2,0].set_title('Mid-plane slice  y=0.5', fontsize=10, fontweight='bold')
ax[2,0].set_xlabel('x'); ax[2,0].set_ylabel('u(x,0.5)'); ax[2,0].legend(fontsize=8); ax[2,0].grid(alpha=.3)

zm = _g >= BL_LEFT - 0.08
ax[2,1].set_facecolor('white')
ax[2,1].plot(_g[zm], u_g[ym][zm], '--', c='#1a1a2e', lw=2.5, label='reference')
ax[2,1].plot(_g[zm], gpr_g[ym][zm], '-', c='#3b6fb0', lw=2, label='plain GPR')
ax[2,1].plot(_g[zm], pi_g[ym][zm], '-', c='#e94560', lw=2, label='PIGP')
ax[2,1].plot(_g[zm], two_g[ym][zm], '-', c='#2a9d3f', lw=2, label='two-stage')
ax[2,1].axvspan(BL_LEFT, 1.0, alpha=.12, color='orange')
ax[2,1].set_title('Boundary-layer zoom  y=0.5', fontsize=10, fontweight='bold')
ax[2,1].set_xlabel('x'); ax[2,1].set_ylabel('u(x,0.5)'); ax[2,1].set_xlim(BL_LEFT-0.08, 1.0)
ax[2,1].legend(fontsize=8); ax[2,1].grid(alpha=.3)

ax[2,2].set_facecolor('white')
ax[2,2].semilogy(_g, egpr_g[ym]+1e-12, c='#3b6fb0', lw=2, label='plain GPR')
ax[2,2].semilogy(_g, epi_g[ym]+1e-12, c='#e94560', lw=2, label='PIGP')
ax[2,2].semilogy(_g, etwo_g[ym]+1e-12, c='#2a9d3f', lw=2, label='two-stage')
ax[2,2].axvspan(BL_LEFT, 1.0, alpha=.08, color='orange')
ax[2,2].set_title('Mid-plane |error|  y=0.5', fontsize=10, fontweight='bold')
ax[2,2].set_xlabel('x'); ax[2,2].set_ylabel('|u-û|'); ax[2,2].legend(fontsize=8); ax[2,2].grid(alpha=.3, which='both')

plt.suptitle(f"2D advection-diffusion  [0,1]²  |  Pe={Pe:.0f}, v=({V_TRUE},0), κ={KAPPA_TRUE}, δ≈{DELTA:.4f}  |  "
             f"forcing='{CASE}'\nplain GPR L2={l2_gpr:.3f}   vs   PIGP L2={l2_pi:.3f}   "
             f"vs   two-stage L2={l2_two:.3f}", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.subplots_adjust(hspace=0.40, wspace=0.35)
plt.savefig(f'advdiff_2d_gpr_pigp_{CASE}.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved: advdiff_2d_gpr_pigp_{CASE}.png")

# ----- dedicated two-stage decomposition figure -------------------------------
res_g = TWO['res'].copy()               # |f - L mu1| on the candidate grid
fig2, ax2 = plt.subplots(2, 3, figsize=(17, 10)); fig2.patch.set_facecolor('#f8f9fa')

def pc2(a, X, Y, d, t, cmap='RdYlBu_r', vmin=None, vmax=None, cb='', scatter=None):
    a.set_facecolor('white')
    if vmin is None: vm = np.nanmax(np.abs(d)); vmin, vmax = -vm, vm
    m = a.pcolormesh(X, Y, d, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    fig2.colorbar(m, ax=a, fraction=0.046, pad=0.04, label=cb)
    if scatter is not None:
        a.scatter(scatter[:,0], scatter[:,1], s=16, c='k', marker='*', lw=0)
    a.axvspan(BL_LEFT, 1.0, alpha=0.08, color='orange')
    a.set_title(t, fontsize=10, fontweight='bold'); a.set_xlabel('x'); a.set_ylabel('y')
    a.set_xlim(0, 1); a.set_ylim(0, 1)

pc2(ax2[0,0], _GX, _GY, u_g, f'reference u(x,y)  [{CASE}]', vmin=-VR, vmax=VR, cb='u')
pc2(ax2[0,1], _GX, _GY, s1_g, f'stage 1: PIGP w/o BCs  (prior mean, L2={relL2(TWO["mu_s1"]):.3f})',
    vmin=-VR, vmax=VR, cb='u')
pc2(ax2[0,2], _GX, _GY, two_g, f'final = stage1 + stage2  (L2={l2_two:.3f})',
    vmin=-VR, vmax=VR, cb='u')

# residual field (on the graded candidate grid) with focused points overlaid
cand = TWO['cand']
ax2[1,0].set_facecolor('white')
sc = ax2[1,0].scatter(cand[:,0], cand[:,1], c=res_g, s=6, cmap='hot_r')
fig2.colorbar(sc, ax=ax2[1,0], fraction=0.046, pad=0.04, label='|f - L(mu1)|')
ax2[1,0].scatter(TWO['xy_focus'][:,0], TWO['xy_focus'][:,1], s=30, c='lime',
                 marker='*', edgecolors='k', lw=.3, label='focused')
ax2[1,0].scatter(TWO['xy_anchor'][:,0], TWO['xy_anchor'][:,1], s=16, c='cyan',
                 marker='D', edgecolors='k', lw=.3, label='anchors')
ax2[1,0].scatter(TWO['xy_bc2'][:,0], TWO['xy_bc2'][:,1], s=8, c='royalblue',
                 marker='.', label='S2 BC (dense)')
ax2[1,0].axvspan(BL_LEFT, 1.0, alpha=.08, color='orange')
ax2[1,0].set_title('stage-1 PDE residual + focused pts', fontsize=10, fontweight='bold')
ax2[1,0].set_xlabel('x'); ax2[1,0].set_ylabel('y'); ax2[1,0].set_xlim(0,1); ax2[1,0].set_ylim(0,1)
ax2[1,0].legend(fontsize=7, loc='upper left')

pc2(ax2[1,1], _GX, _GY, dl_g, 'stage-2 correction  δu', cb='δu',
    scatter=TWO['xy_focus'])
pc2(ax2[1,2], _GX, _GY, etwo_g, 'final |error|', cmap='hot_r', vmin=0, vmax=EMAX, cb='|u-û|')

plt.suptitle(f"Two-stage residual correction  |  stage1 L2={relL2(TWO['mu_s1']):.3f}  ->  "
             f"stage1+stage2 L2={l2_two:.3f}   ({relL2(TWO['mu_s1'])/l2_two:.2f}× overall)",
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.subplots_adjust(hspace=0.30, wspace=0.30)
plt.savefig(f'advdiff_2d_twostage_{CASE}.png', dpi=150, bbox_inches='tight')
print(f"Figure saved: advdiff_2d_twostage_{CASE}.png")

if SHOW_3D:
    import plotly.graph_objects as go, plotly.io as pio
    pio.renderers.default = "browser"
    for z, ttl in [(two_g, 'two-stage posterior mean'), (pi_g, 'PIGP posterior mean'), (u_g, 'reference')]:
        f3 = go.Figure(go.Surface(x=_GX, y=_GY, z=z, colorscale='Viridis'))
        f3.update_layout(title=ttl, scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='u'))
        f3.show()
print("Done.")


NOISE_LEVELS = [1e-4,0.5e-3,1e-3,0.5e-2, 1e-2,0.5e-1, 1e-1]
errors_PIGP = []
errors_GPR = []
errors_TWO = []
SNRs = []

for nl in NOISE_LEVELS:
    print(f"\n\n=== NOISE_OBS = {nl:.1e} ===")
    u_obs  = ref(xy_obs) + np.random.normal(0, nl, N_OBS)
    u_obs_n = u_obs/S
    SN_FIX = nl / S
    print(f"Observation range: [{np.min(u_obs):.3f}, {np.max(u_obs):.3f}]")
    print(f"Standardised noise: {SN_FIX:.3e}")
    GPR  = fit(False, np.array([0.15, 0.15, 1.0]))
    PIGP = fit(True,  np.array([0.3, 0.3, 1.0]))
    TWO  = fit_two_stage(np.array([0.3, 0.3, 1.0]))
    errors_GPR.append(relL2(GPR['mu']))
    errors_PIGP.append(relL2(PIGP['mu']))
    errors_TWO.append(relL2(TWO['mu']))
    SNRs.append(np.mean(u_obs**2)/nl**2)

fig = plt.figure(figsize=(6, 4))
plt.plot(NOISE_LEVELS, errors_GPR, 'o-', label='plain GPR', color='#3b6fb0')
plt.plot(NOISE_LEVELS, errors_PIGP, 's-', label='PIGP', color='#e94560')
plt.plot(NOISE_LEVELS, errors_TWO, '^-', label='two-stage', color='#2a9d3f')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Observation noise level (std dev)'); plt.ylabel('Relative L2 error')
plt.grid(which='both', alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(f'advdiff_2d_gpr_pigp_{CASE}_noise_sensitivity.png', dpi=150)
plt.show()


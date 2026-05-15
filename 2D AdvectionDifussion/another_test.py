import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import minimize
from numpy.linalg import slogdet


import numpy.random as npr


def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)




def log_prior(theta):
    l, sigma, log_sigma_n, v, kappa = theta

    log_l = np.log(l)
    log_sigma = np.log(sigma)

    mu_l, std_l = np.log(0.3), 0.5
    mu_s, std_s = np.log(1.0), 0.5
    mu_n, std_n = np.log(1e-2), 1.0
    mu_v, std_v = 0.5, 0.5
    mu_k, std_k = 0.5, 0.5

    lp  = -0.5 * ((log_l - mu_l)/std_l)**2
    lp += -0.5 * ((log_sigma - mu_s)/std_s)**2
    lp += -0.5 * ((log_sigma_n - mu_n)/std_n)**2
    lp += -0.5 * ((v - mu_v)/std_v)**2
    lp += -0.5 * ((kappa - mu_k)/std_k)**2

    return lp

def build_L(obs_grid, model_grid):
    n = len(obs_grid)
    m = len(model_grid)
    L = np.zeros((n, m))
    for i, obs_pt in enumerate(obs_grid):
        j = np.searchsorted(model_grid, obs_pt) - 1
        j = np.clip(j, 0, m - 2)
        t = (obs_pt - model_grid[j]) / (model_grid[j+1] - model_grid[j])
        L[i, j]     = 1 - t
        L[i, j + 1] = t
    return L

a, b = 0.0, 1.0
n1, n2 = 100, 100
m1, m2 = 30, 30
N = n1 * n2  # observation space dimension

x = np.linspace(a, b, n1)
y = np.linspace(a, b, n2)

Xg, Yg = np.meshgrid(x, y, indexing="ij")
yf = f(Xg, Yg).ravel()

ux = np.linspace(a, b, m1)
uy = np.linspace(a, b, m2)

hx = ux[1] - ux[0]
hy = uy[1] - uy[0]

XX, YY = np.meshgrid(ux, uy, indexing="ij")

Lx = build_L(x, ux)  # from model grid ux to observation grid x
Ly = build_L(y, uy)  # from model grid uy to observation grid y

def rbf_kernel_1d(x, l, sigma):
    sqdist = (x[:, None] - x[None, :])**2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)


def kron_mv(Kx, Ky, vec):
    V = vec.reshape(m1, m2)
    KV = Kx @ V @ Ky.T
    return KV.ravel()



def build_K_1d(obs_grid, model_grid, h, l, sigma, sigma_n, v, kappa):
    """
    Build kernel on observation grid, interpolating from model grid.
    obs_grid: n observation points
    model_grid: m model grid points
    Returns: K (n×n kernel on observations), L (n×m interpolation matrix)
    """
    n = len(obs_grid)
    m = len(model_grid)
    
    # Kernel on model grid (m × m)
    Ku = rbf_kernel_1d(model_grid, l, sigma)
    Ku += 1e-6 * np.eye(m)  # jitter

    # Interpolation matrix from model grid to observation points (n × m)
    L = np.zeros((n, m))
    for i, obs_pt in enumerate(obs_grid):
        j = np.searchsorted(model_grid, obs_pt) - 1
        j = np.clip(j, 0, m - 2)
        t = (obs_pt - model_grid[j]) / (model_grid[j+1] - model_grid[j])
        L[i, j]     = 1 - t
        L[i, j + 1] = t

    # Differential operators on model grid (m × m)
    D1 = np.zeros((m, m))
    D2 = np.zeros((m, m))

    for i in range(1, m - 1):
        D1[i, i-1] = -1/(2*h)
        D1[i, i+1] =  1/(2*h)

        D2[i, i-1] =  1/(h**2)
        D2[i, i]   = -2/(h**2)
        D2[i, i+1] =  1/(h**2)

    # PDE operator on model grid (m × m)
    Op = v * D1 - kappa * D2
    
    # Kernel on observations: K = L @ Op @ Ku @ Op.T @ L.T (n × m @ m × m @ m × m @ m × m @ m × n = n × n)
    K = L @ Op @ Ku @ Op.T @ L.T
    K += sigma_n**2 * np.eye(n)

    return K, L




def nlml(theta):
    l, sigma, log_sigma_n, v, kappa = theta
    sigma_n = np.exp(log_sigma_n)

    Kx, Lx = build_K_1d(x, ux, hx, l, sigma, sigma_n, v, kappa)
    Ky, Ly = build_K_1d(y, uy, hy, l, sigma, sigma_n, v, kappa)

    def mv(vec):
        return kron_mv(Kx, Ky, vec)

    K = LinearOperator((N, N), matvec=mv)
    
    alpha, info = cg(K, yf, rtol=1e-6, maxiter=3000)
    if info != 0:
        raise RuntimeError("CG did not converge")

    sign_x, logdet_x = slogdet(Kx)
    sign_y, logdet_y = slogdet(Ky)

    logdet = n2 * logdet_x + n1 * logdet_y

    return 0.5 * yf @ alpha + 0.5 * logdet


def nlp(theta):
    return nlml(theta) - log_prior(theta)




theta0 = np.array([0.2, 1.0, np.log(1e-2), 0.5, 0.5])

bounds = [
    (0.05, 1.0),
    (0.1, 5.0),
    (-10, 0),
    (0.0, 1.0),
    (0.0, 1.0)
]

#res = minimize(nlp, theta0, method="L-BFGS-B",
#               bounds=bounds, options={"maxiter": 100})

#l_opt, sigma_opt, log_sigma_n_opt, v_opt, kappa_opt = res.x
#sigma_n_opt = np.exp(log_sigma_n_opt)
'''
print("\nOptimized hyperparameters:")
print(f"l = {l_opt:.4f}")
print(f"sigma = {sigma_opt:.4f}")
print(f"sigma_n = {sigma_n_opt:.4e}")
print(f"v = {v_opt:.4f}")
print(f"kappa = {kappa_opt:.4f}")




Kx, Lx = build_K_1d(x, ux, hx, l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt)
Ky, Ly = build_K_1d(y, uy, hy, l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt)
'''
#K = LinearOperator((N, N), matvec=lambda vec: kron_mv(Kx, Ky, vec))

#alpha, info = cg(K, yf, rtol=1e-6, maxiter=3000)

# Reconstruct solution on observation grid
#u_mean_grid = alpha.reshape(n1, n2)



#fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#im1 = axes[0].contourf(Xg, Yg, u_mean_grid, levels=20)
#axes[0].set_title("Estimated Solution")

#im2 = axes[1].contourf(Xg, Yg, f(Xg, Yg), levels=20)
#axes[1].set_title("True Solution")

#plt.tight_layout()
#plt.show()



# ============================================================
# ========== CORRECTED KISSGP PDE IMPLEMENTATION ============
# ============================================================

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
import numpy.random as npr


# ------------------------------------------------------------
# 1. Build pure kernel (NO NOISE, NO PDE INSIDE)
# ------------------------------------------------------------

def build_pure_K_1d(grid, l, sigma):
    K = rbf_kernel_1d(grid, l, sigma)
    K += 1e-6 * np.eye(len(grid))
    return K


# ------------------------------------------------------------
# 2. Build differential operators (1D)
# ------------------------------------------------------------

def build_diff_ops(m, h):
    D1 = np.zeros((m, m))
    D2 = np.zeros((m, m))

    for i in range(1, m - 1):
        D1[i, i-1] = -1/(2*h)
        D1[i, i+1] =  1/(2*h)

        D2[i, i-1] =  1/(h**2)
        D2[i, i]   = -2/(h**2)
        D2[i, i+1] =  1/(h**2)

    return D1, D2


Dx, Dxx = build_diff_ops(m1, hx)
Dy, Dyy = build_diff_ops(m2, hy)


# ------------------------------------------------------------
# 3. Correct 2D PDE operator application (implicit)
#    L = v_x ∂x + v_y ∂y - κ (∂xx + ∂yy)
# ------------------------------------------------------------

def apply_L(vec, v, kappa):
    U = vec.reshape(m1, m2)

    term_x  = v * (Dx @ U)
    term_y  = v * (U @ Dy.T)
    term_xx = kappa * (Dxx @ U)
    term_yy = kappa * (U @ Dyy.T)

    return (term_x + term_y - term_xx - term_yy).ravel()


def apply_Lt(vec, v, kappa):
    # adjoint operator
    U = vec.reshape(m1, m2)

    term_x  = -v * (Dx.T @ U)
    term_y  = -v * (U @ Dy)
    term_xx = -kappa * (Dxx.T @ U)
    term_yy = -kappa * (U @ Dyy)

    return (term_x + term_y - term_xx - term_yy).ravel()


# ------------------------------------------------------------
# 4. Build corrected NLML
# ------------------------------------------------------------

def nlml_correct(theta):

    log_l, log_sigma, log_sigma_n, v, log_kappa = theta
    l = np.exp(log_l)
    sigma = np.exp(log_sigma)
    sigma_n = np.exp(log_sigma_n)
    kappa = np.exp(log_kappa)

    # Pure kernel on inducing grid
    Kx = build_pure_K_1d(ux, l, sigma)
    Ky = build_pure_K_1d(uy, l, sigma)

    def mv(vec):

        # Interpolate to inducing grid
        u = vec.reshape(n1, n2)

        # Map to inducing space
        u_ind = Lx.T @ u @ Ly
        u_ind = u_ind.ravel()

        # Apply L^T
        tmp = apply_Lt(u_ind, v, kappa)

        # Apply kernel (Kronecker)
        tmp = kron_mv(Kx, Ky, tmp)

        # Apply L
        tmp = apply_L(tmp, v, kappa)

        tmp = tmp.reshape(m1, m2)

        # Interpolate back to observation grid
        tmp = Lx @ tmp @ Ly.T
        tmp = tmp.ravel()

        # Add noise OUTSIDE kronecker
        return tmp + sigma_n**2 * vec

    K = LinearOperator((N, N), matvec=mv)

    alpha, info = cg(K, yf, rtol=1e-6, maxiter=2000)
    if info != 0:
        raise RuntimeError("CG failed")

    # -------------------------
    # Stochastic Lanczos logdet
    # -------------------------

    S = 10
    logdet_est = 0.0

    for _ in range(S):
        z = npr.randn(N)
        z /= np.linalg.norm(z)

        w, _ = cg(K, z, rtol=1e-6, maxiter=1000)
        logdet_est += z @ w

    logdet_est *= N / S

    return 0.5 * yf @ alpha + 0.5 * logdet_est


# ------------------------------------------------------------
# 5. Optimize corrected objective
# ------------------------------------------------------------

theta0_corr = np.log([0.3, 1.0, 1e-2, 0.5])
theta0_corr = np.concatenate([theta0_corr[:3], [0.5], theta0_corr[3:]])

bounds_corr = [
    (-3, 1),   # log l
    (-3, 3),   # log sigma
    (-10, 0),  # log noise
    (0, 2),    # v
    (-5, 2)    # log kappa
]

res_corr = minimize(nlml_correct, theta0_corr,
                    method="L-BFGS-B",
                    bounds=bounds_corr,
                    options={"maxiter": 50})

print("\nCorrected Optimization:")
print(res_corr)


# ------------------------------------------------------------
# 6. Correct posterior mean of u
# ------------------------------------------------------------

theta_opt = res_corr.x
log_l, log_sigma, log_sigma_n, v_opt, log_kappa = theta_opt
l = np.exp(log_l)
sigma = np.exp(log_sigma)
sigma_n = np.exp(log_sigma_n)
kappa = np.exp(log_kappa)

Kx = build_pure_K_1d(ux, l, sigma)
Ky = build_pure_K_1d(uy, l, sigma)

def mv_final(vec):

    u = vec.reshape(n1, n2)
    u_ind = Lx.T @ u @ Ly
    u_ind = u_ind.ravel()

    tmp = apply_Lt(u_ind, v_opt, kappa)
    tmp = kron_mv(Kx, Ky, tmp)
    tmp = apply_L(tmp, v_opt, kappa)

    tmp = tmp.reshape(m1, m2)
    tmp = Lx @ tmp @ Ly.T
    tmp = tmp.ravel()

    return tmp + sigma_n**2 * vec

K_corr = LinearOperator((N, N), matvec=mv_final)

alpha, _ = cg(K_corr, yf, rtol=1e-6, maxiter=2000)

# Proper posterior mean:
# μ_u = K_u L^T K^{-1} y

u_ind = Lx.T @ alpha.reshape(n1, n2) @ Ly
u_ind = apply_Lt(u_ind.ravel(), v_opt, kappa)
u_ind = kron_mv(Kx, Ky, u_ind)

u_mean = u_ind.reshape(m1, m2)
u_mean = Lx @ u_mean @ Ly.T

# ------------------------------------------------------------
# Plot corrected solution
# ------------------------------------------------------------

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(Xg, Yg, u_mean, levels=20)
plt.title("Corrected Posterior Mean")

plt.subplot(1,2,2)
plt.contourf(Xg, Yg, f(Xg, Yg), levels=20)
plt.title("True Solution")

plt.tight_layout()
plt.show()

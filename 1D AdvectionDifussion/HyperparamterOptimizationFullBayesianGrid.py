# --- Full Bayesian integration on a discretized grid ---
# Assumes: x, c, y, x_star, nlp, log_prior, build_K, build_K_star are defined. :contentReference[oaicite:2]{index=2}

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize
import time

def u_exact(x, kappa, mu):
    import numpy as np
    pi = np.pi

    denom = pi**2 * kappa**2 + mu**2

    A = kappa / denom
    B = -mu / (np.pi * denom)

    # homogeneous part exponent
    r = mu / kappa

    #  C1, C2 using boundary conditions
    # u(0)=0
    # u(1)=0

    # At x=0:
    # C1 + C2 + B = 0

    # At x=1:
    # C1 + C2*exp(r) + A sin(pi) + B cos(pi) = 0
    # sin(pi)=0, cos(pi)=-1

    # => C1 + C2 e^r - B = 0

    e = np.exp(r)
    C2 = (2*B) / (e - 1)
    C1 = -B - C2

    return (
        C1
        + C2*np.exp(r*x)
        + A*np.sin(pi*x)
        + B*np.cos(pi*x)
    )

def log_prior(theta):
    l, sigma, log_sigma_n,v,kappa = theta

    log_l = np.log(l)
    log_sigma = np.log(sigma) #force positivity of hyperparameters

    # Hyperprior parameters
    #the bounds are made wrt to the fact that the domain is [0,1] and the expected behavior of the equation

    mu_l, std_l = np.log(0.3), 0.8 #multiplicative deviations
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

def build_K_star(x_star, x, c, l, sigma, v, kappa):
    xb = x[:2]
    xu = x[2:]
    xf = c 
    x = np.hstack([xb, xu])  # ensure x is ordered with boundaries first, check different ordering
    kbb, *_ = kernel_derivatives(x_star, x , l, sigma)

    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x_star, xf, l, sigma)
    Kuf = v * k_xp - kappa * k_xpxp

    return np.hstack([kbb, Kuf])

def kernel_derivatives(x,y,l,sigma):
    r = x[:, None] - y[None, :]
    k = sigma**2 * np.exp(-0.5 * r**2 / l**2)

    l2 = l**2
    l4 = l**4
    l6 = l**6
    l8 = l**8

    k_x  = -(r/l2) * k
    k_xp = +(r/l2) * k

    k_xx  = (r**2/l4 - 1/l2) * k
    k_xpxp = k_xx
    k_xxp = (1/l2 - r**2/l4) * k

    k_xxxp = (-3*r/l4 + r**3/l6) * k # -3/l4 r+1/l6 r^3 
    #k_xxpxp = (3*r**2/l6 - 1/l4) * k # 3/l4r-r^3/l6
    k_xxpxp = (3*r/l4 - r**3/l6) * k # 3/l4r-r^3/l6

    k_xxxpxp = (3/l4 - 6*r**2/l6 + r**4/l8) * k
    
    #if (k_xxxp.shape[0] == k_xxxp.shape[1]):
    #    print('Symmetry check (k_xxxp - k_xxxp.T):', np.max(np.abs(k_xxxp - k_xxxp.T)))
    #    print('Symmetry check (k_xxpxp - k_xxpxp.T):', np.max(np.abs(k_xxpxp - k_xxpxp.T)))
    #    print('Symmetry check (k_xxxpxp - k_xxxpxp.T):', np.max(np.abs(k_xxxpxp - k_xxxpxp.T)))

    return k, k_x, k_xp, k_xx, k_xpxp, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp



def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

def f(x):
    return np.sin(np.pi * x)
def nlp(theta):
    return nlml(x,theta, with_bc=True) - log_prior(theta)

def build_K(x,c,l,sigma,sigma_n,v,kappa,with_bc=False):
    
    xb = x[:2]  # boundary points
    xu = x[2:]  # interior points
    xf = c
    nf = len(xu)
    x = np.hstack([xb, xu])  # ensure x is ordered with boundaries first, check different ordering 
    nb = len(x)  

    nxc = len(x) + len(c)
    K = np.zeros((nxc, nxc))


    kbb,*_ = kernel_derivatives(x, x, l, sigma)
    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x, xf, l, sigma)

    Kbf = v * k_xp - kappa * k_xpxp
    K[:nb,:nb]  = kbb
    K[:nb, nb:] = Kbf

    
    kfb, k_x, _, k_xx, *_ = kernel_derivatives(xf, x, l, sigma)

    Kfb = v * k_x - kappa * k_xx
    K[nb:, :nb] = Kfb

    (_, _, _, _, _, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp) = kernel_derivatives(xf, xf, l, sigma)

    #Kff = v**2 * k_xxp - v*kappa * k_xxxp - v*kappa * k_xxpxp + kappa**2 * k_xxxpxp
    # kxxxp = - kxxpxp
    Kff = v**2 * k_xxp + kappa**2 * k_xxxpxp
    Kff += sigma_n**2 * np.eye(len(c))

    K[nb:, nb:] = Kff

    if not with_bc:
        return K[2:,2:]
    
    return K

def nlml(x,theta,with_bc=False):
    l, sigma, log_sigma_n,v,kappa = theta
    sigma_n = np.exp(log_sigma_n)
    
    K = build_K(x,c,l,sigma,sigma_n,v,kappa,with_bc) 
    K += 1e-6 * np.eye(K.shape[0]) 
    sign, logdet = slogdet(K)

    #alpha, info = cg(K, y, rtol=1e-8, maxiter=5000)
    alpha = solve(K, y)
    return 0.5 * y.T @ alpha + 0.5 * logdet

a,b = 0.0, 5.0
n = 100
m = 70
nt = 70
nc = 10
x= np.sort(np.random.uniform(a, b, n-2)) #n-2
c = np.sort(np.random.uniform(a, b, nc))
u = np.linspace(a, b, m)
#t = np.random.uniform(a, b, nt)
x_star = np.sort(np.random.uniform(a, b, 30))
h = u[1] - u[0]
#idx_f = np.arange(0,n)
idx_u = np.arange(0,m)
#yt = f(x_star)
#yf = f(x) + np.random.normal(0, 0.001, size=len(x))
kappa_true = 0.6
v_true = 0.3


yt = u_exact(x_star, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x_star))
yu = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))

yb =  np.array([0.0, 0.0])
#y = np.hstack([yu, yf])
x = np.hstack([a,b, x])  # add boundary points
yf = f(c) + np.random.normal(0, 0.1, size=len(c))
y = np.hstack([yb, yu, yf])


l = 0.2
sigma = 0.3
sigma_opt = 0.6
sigma_n_opt = 1e-3
v_opt = 0.5
sigma_n = 1e-2
v = 0.8
kappa = 0.2

Kxx_prior = rbf_kernel(x, x, l, sigma)
prior_mean = np.zeros_like(x)
prior_std = np.sqrt(np.diag(Kxx_prior))

def grid_bayesian_integration(
    l_vals, kappa_vals,
    sigma_fixed= sigma_opt if 'sigma_opt' in globals() else 0.3,
    sigma_n_fixed = (sigma_n_opt if 'sigma_n_opt' in globals() else 1e-2),
    v_fixed = (v_opt if 'v_opt' in globals() else 0.5),
    with_bc=True,
    jitter=1e-10
):
    # Prepare storage
    L, K = np.meshgrid(l_vals, kappa_vals, indexing='xy')
    grid_shape = L.shape
    n_grid = L.size

    log_post = np.empty(n_grid)    # log posterior (unnormalized)
    preds = np.zeros((n_grid, len(x_star)))   # predictive mean for each grid point
    pred_vars = np.zeros((n_grid, len(x_star)))  # predictive diag variance (optional)
    

    # Create full grid for all hyperparameters
    L, K, S, SN, V = np.meshgrid(
        l_vals, kappa_vals, sigma_vals, sigma_n_vals, v_vals, indexing='ij'
    )
    grid_shape = L.shape
    n_grid = L.size

    log_post = np.empty(n_grid)
    preds = np.zeros((n_grid, len(x_star)))
    pred_vars = np.zeros((n_grid, len(x_star)))

    # flatten iteration
    it = 0
    for idx in np.ndindex(grid_shape):
        l_val = float(L[idx])
        kappa_val = float(K[idx])
        sigma_val = float(S[idx])
        sigma_n_val = float(SN[idx])
        v_val = float(V[idx])

        theta = np.array([l_val, sigma_val, np.log(sigma_n_val), v_val, kappa_val])
        nlp_val = nlp(theta)
        log_post[it] = - nlp_val

        nxc = len(x) + len(c)
        Kxx = build_K(x, c, l_val, sigma_val, sigma_n_val, v_val, kappa_val, with_bc=with_bc)
        Kxx += jitter * np.eye(Kxx.shape[0])

        try:
            alpha = solve(Kxx, y)
        except np.linalg.LinAlgError:
            Kxx += 1e-6 * np.eye(Kxx.shape[0])
            alpha = solve(Kxx, y)

        Kuz = build_K_star(x_star, x, c, l_val, sigma_val, v_val, kappa_val)
        mu_star = Kuz @ alpha
        preds[it, :] = mu_star

        Kxx_inv_KuzT = solve(Kxx, Kuz.T)
        Kuu = rbf_kernel(x_star, x_star, l_val, sigma_val)
        pred_cov_diag = np.diag(Kuu) - np.sum(Kuz * (Kxx_inv_KuzT.T), axis=1)
        pred_vars[it, :] = np.maximum(pred_cov_diag, 0.0)

        it += 1

    # Normalize weights in log-domain
    max_log = np.max(log_post)
    weights = np.exp(log_post - max_log)
    weights[weights < 1e-12] = 0.0
    weights /= np.sum(weights)

    post_mean = weights @ preds
    E_var = weights @ pred_vars
    E_mean_sq = weights @ (preds**2)
    Var_mean = E_mean_sq - post_mean**2
    post_var = E_var + Var_mean

    weight_grid = weights.reshape(grid_shape)

    return {
        "post_mean": post_mean,
        "post_var": post_var,
        "weights": weight_grid,
        "all_preds": preds,
        "all_pred_vars": pred_vars,
        "grid_L": L,
        "grid_K": K,
        "grid_sigma": S,
        "grid_sigma_n": SN,
        "grid_v": V
    }
    # flatten iteration
    it = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            l_val = float(L[i, j])
            kappa_val = float(K[i, j])

            theta = np.array([l_val, sigma_fixed, np.log(sigma_n_fixed), v_fixed, kappa_val])
            nlp_val = nlp(theta)    
            log_post[it] = - nlp_val

            nxc = len(x) + len(c)
            Kxx = build_K(x, c, l_val, sigma_fixed, sigma_n_fixed, v_fixed, kappa_val, with_bc=with_bc)
            Kxx += jitter * np.eye(Kxx.shape[0])

            try:
                alpha = solve(Kxx, y)
            except np.linalg.LinAlgError:
            
                Kxx += 1e-6 * np.eye(Kxx.shape[0])
                alpha = solve(Kxx, y)

            Kuz = build_K_star(x_star, x, c, l_val, sigma_fixed, v_fixed, kappa_val)

            mu_star = Kuz @ alpha
            preds[it, :] = mu_star

            
            Kxx_inv_KuzT = solve(Kxx, Kuz.T)   
           
            
            Kuu = rbf_kernel(x_star, x_star, l_val, sigma_fixed)
            pred_cov_diag = np.diag(Kuu) - np.sum(Kuz * (Kxx_inv_KuzT.T), axis=1)
            pred_vars[it, :] = np.maximum(pred_cov_diag, 0.0)  # numerical safety

            it += 1

    # Normalize weights in log-domain
    max_log = np.max(log_post)
    weights = np.exp(log_post - max_log)
    weights[weights < 1e-12] = 0.0
    weights /= np.sum(weights)

    # weighted predictive mean and variance
    post_mean = weights @ preds     # shape: (n_star,)
    # law of total variance: Var = E[var|theta] + Var[E[f|theta]]
    E_var = weights @ pred_vars
    E_mean_sq = weights @ (preds**2)
    Var_mean = E_mean_sq - post_mean**2
    post_var = E_var + Var_mean

    # reshape weights to grid shape for inspection
    weight_grid = weights.reshape(grid_shape)

    return {
        "post_mean": post_mean,
        "post_var": post_var,
        "weights": weight_grid,
        "all_preds": preds,
        "all_pred_vars": pred_vars,
        "grid_L": L,
        "grid_K": K
    }


l_vals = np.linspace(0.08, 1.2, 10)         
kappa_vals = np.linspace(0.05, 1.0, 10)
sigma_vals = np.linspace(0.1, 1.0, 10)
sigma_n_vals = np.linspace(1e-3, 1e-1, 10)
v_vals = np.linspace(0.1, 1.0, 10)     
grid_out = grid_bayesian_integration(l_vals, kappa_vals,sigma_vals
                                    ,sigma_n_vals,v_vals, 
                                     with_bc=True)

print("Posterior mean shape:", grid_out["post_mean"].shape)

# Posterior predictive mean at x_star available as grid_out["post_mean"]
#plot posterior with uncertainty range
plt.figure(figsize=(10, 6))
plt.plot(x_star, grid_out["post_mean"], label="Posterior Mean")
plt.fill_between(x_star, 
                 grid_out["post_mean"] - 2*np.sqrt(grid_out["post_var"]), 
                 grid_out["post_mean"] + 2*np.sqrt(grid_out["post_var"]), 
                 color='lightblue', alpha=0.5, label="Posterior ±2 std")
plt.scatter(x, y[:len(x)], color='red', label="Observations", zorder=1)
plt.scatter(x_star, yt, color='green', label="Test points (x_star)", zorder=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Full Bayesian Hyperparameter Integration")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(grid_out["weights"], origin="lower")
plt.colorbar()
plt.title("Hyperparameter posterior")
plt.xlabel("l index")
plt.ylabel("kappa index")
plt.show()
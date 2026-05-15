

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from numpy.linalg import solve, slogdet
import time

def u_exact(x, kappa, mu):
    import numpy as np
    pi = np.pi

    denom = pi**2 * kappa**2 + mu**2

    A = kappa / denom
    B = -mu / (np.pi * denom)

    r = mu / kappa



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
    
    eps = 1e-8
    log_l = np.log(np.maximum(l, eps))
    log_sigma = np.log(np.maximum(sigma, eps)) 


    mu_l, std_l = np.log(0.3), 0.8 
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
    x = x[:n]  
    xf = c 
    kbb, *_ = kernel_derivatives(x_star, x , l, sigma)

    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x_star, xf, l, sigma)
    Kuf = v * k_xp - kappa * k_xpxp

    return np.hstack([kbb, Kuf])

def kernel_derivatives(x, y, l, sigma):
  
    min_l = 0.01  # Enforce minimum lengthscale
    if l < min_l:
        print(f"Warning: l={l:.6e} too small, clamping to {min_l}")
        l = min_l
    
    r = x[:, None] - y[None, :]
    k = sigma**2 * np.exp(-0.5 * r**2 / l**2)

    l2 = l**2
    l4 = l**4
    l6 = l**6
    l8 = l**8

    k_x = -(r / l2) * k
    k_xp = +(r / l2) * k

    k_xx = (r**2 / l4 - 1 / l2) * k
    k_xpxp = k_xx
    k_xxp = (1 / l2 - r**2 / l4) * k

    k_xxxp = (-3*r/l4 + r**3/l6) * k 
    #k_xxpxp = (3*r**2/l6 - 1/l4) * k # 3/l4r-r^3/l6
    k_xxpxp = (3*r/l4 - r**3/l6) * k # 3/l4r-r^3/l6

    k_xxxpxp = (3/l4 - 6*r**2/l6 + r**4/l8) * k
    
    # Detect and handle NaN
    result = (k, k_x, k_xp, k_xx, k_xpxp, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp)
    if any(np.any(~np.isfinite(arr)) for arr in result):
        print(f"Warning: NaN detected in kernel_derivatives with l={l}, sigma={sigma}")
        print(f"  r range: [{np.min(r):.3f}, {np.max(r):.3f}]")
        print(f"  k range: [{np.min(k[np.isfinite(k)]):.3e}, {np.max(k[np.isfinite(k)]):.3e}]")

    return result



def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

def f(x):
    return np.sin(np.pi * x)

def nlp(theta):
    return nlml(x,theta, with_bc=True) - log_prior(theta)

def build_K(x,c,l,sigma,sigma_n,v,kappa,with_bc=True):
    
    
    xf = c
    x = x[:n]  # ensure x is ordered with boundaries first, check different ordering 
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

def nlml(x,theta,with_bc=True):
    l, sigma, log_sigma_n,v,kappa = theta
    sigma_n = np.exp(log_sigma_n)
    
    K = build_K(x,c,l,sigma,sigma_n,v,kappa,with_bc) 
    K += 1e-6 * np.eye(K.shape[0]) 
    sign, logdet = slogdet(K)

    # Check for NaN/inf in logdet
    if not np.isfinite(logdet):
        return 1e10  # Return large penalty for invalid covariance
    
    #alpha, info = cg(K, y, rtol=1e-8, maxiter=5000)
    try:
        alpha = solve(K, y)
    except np.linalg.LinAlgError:
        return 1e10 
    
    result = 0.5 * y.T @ alpha + 0.5 * logdet
    
    # Check for NaN/inf in result
    if not np.isfinite(result):
        return 1e10
    
    return result

class VariationalGaussian:
  
    def __init__(self, dim, init_mu, init_logsigma):
        self.dim = dim
        self.mu = np.array(init_mu, dtype=float)
        self.logsigma = np.array(init_logsigma, dtype=float)

    def sample(self, n_samples):
        eps = np.random.randn(n_samples, self.dim)
        samples = self.mu + np.exp(self.logsigma) * eps
        return samples

    def log_prob(self, samples):
        sigma = np.exp(self.logsigma)
        norm_constant = -0.5 * self.dim * np.log(2 * np.pi) - np.sum(self.logsigma)
        sq_diff = ((samples - self.mu) / sigma) ** 2
        return norm_constant - 0.5 * np.sum(sq_diff, axis=1)

    def grad_log_prob(self, samples):
   
        sigma = np.exp(self.logsigma)
        dev = (samples - self.mu) / sigma          # standardized deviation
        grad_mu = dev / sigma                       # ∂ log q / ∂μ
        grad_logsigma = -1 + dev**2                 # ∂ log q / ∂logσ
        return grad_mu, grad_logsigma

def variational_inference(x, c, y, x_star,
                          init_theta,          # [l, sigma, sigma_n, v, kappa]
                          with_bc=True,
                          jitter=1e-10,
                          lr=0.01,
                          lr_decay=0.995,
                          n_iter=100,
                          batch_size=50,
                          n_pred_samples=1000):

    # Transform initial hyperparameters to unconstrained space
    # Order: log(l), log(sigma), log(sigma_n), v, kappa
    init_natural = np.array([
        np.log(init_theta[0]),
        np.log(init_theta[1]),
        np.log(init_theta[2]),
        init_theta[3],
        init_theta[4]
    ])

    var_dist = VariationalGaussian(dim=5,
                                    init_mu=init_natural,
                                    init_logsigma=np.zeros(5))

    elbo_vals = []

    # Moving average baseline for variance reduction
    baseline = 0.0
    beta = 0.9  # smoothing factor

    print("Starting variational inference...")
    for it in range(n_iter):
        eta = var_dist.sample(batch_size)      # shape: (batch_size, 5)

        # Compute log joint and log q for each sample
        log_joint = np.zeros(batch_size)
        for i in range(batch_size):
            # Map back to original hyperparameters with clipping
            # Clamp eta to reasonable ranges to prevent extreme values
            eta_clamped = np.array(eta[i])
            eta_clamped[0] = np.clip(eta_clamped[0], -3, 2)   # log(l): exp(-3) ≈ 0.05 to exp(2) ≈ 7.4 
            eta_clamped[1] = np.clip(eta_clamped[1], -3, 1)   # log(sigma): exp(-3) to exp(1)
            eta_clamped[2] = np.clip(eta_clamped[2], -8, 2)   # log(sigma_n): exp(-8) to exp(2)
            eta_clamped[3] = np.clip(eta_clamped[3], 0.01, 2) # v: bounded in (0, 2)
            eta_clamped[4] = np.clip(eta_clamped[4], 0.01, 2) # kappa: bounded in (0, 2)
            
            l = np.exp(eta_clamped[0])
            sigma = np.exp(eta_clamped[1])
            sigma_n = np.exp(eta_clamped[2])
            v = eta_clamped[3]
            kappa = eta_clamped[4]

            theta = np.array([l, sigma, sigma_n, v, kappa])  
            # log p(y|θ) = -nlml(θ)   (nlml returns negative log marginal likelihood)
            # log p(θ)   = log_prior(θ)
            log_joint[i] = -nlml(x, theta, with_bc=with_bc) + log_prior(theta)

        log_q = var_dist.log_prob(eta)          # (batch_size,)

        # ELBO sample = log p(y,θ) - log q(θ)
        f_elbo = log_joint - log_q
        elbo_mean = np.mean(f_elbo)
        elbo_vals.append(elbo_mean)

        # Update baseline (exponential moving average)
        baseline = beta * baseline + (1 - beta) * elbo_mean

        grad_mu, grad_logsigma = var_dist.grad_log_prob(eta)

        # Score‑function gradient estimator (with baseline)
        grad_mu_avg = np.mean((f_elbo - baseline)[:, None] * grad_mu, axis=0)
        grad_logsigma_avg = np.mean((f_elbo - baseline)[:, None] * grad_logsigma, axis=0)

        # Gradient ascent update
        var_dist.mu += lr * grad_mu_avg
        var_dist.logsigma += lr * grad_logsigma_avg

        # Optional: clip logsigma to avoid extreme values
        var_dist.logsigma = np.clip(var_dist.logsigma, -5, 2)

        if it % 50 == 0:
            print(f"Iter {it:4d}, ELBO = {elbo_mean:.4f}")

    print("Optimisation finished.\n")

    # Sample from final variational distribution
    eta_samples = var_dist.sample(n_pred_samples)

    n_star = len(x_star)
    pred_means = np.zeros((n_pred_samples, n_star))
    pred_vars  = np.zeros((n_pred_samples, n_star))

    for i in range(n_pred_samples):
        l = np.exp(eta_samples[i, 0])
        sigma = np.exp(eta_samples[i, 1])
        sigma_n = np.exp(eta_samples[i, 2])
        v = eta_samples[i, 3]
        kappa = eta_samples[i, 4]

        # Build full covariance and solve
        Kxx = build_K(x, c, l, sigma, sigma_n, v, kappa, with_bc=with_bc)
        Kxx += jitter * np.eye(Kxx.shape[0])
        try:
            print(Kxx.shape,y.shape)
            alpha = solve(Kxx, y)
        except np.linalg.LinAlgError:
            Kxx += 1e-6 * np.eye(Kxx.shape[0])
            alpha = solve(Kxx, y)

        # Predictive mean
        Kuz = build_K_star(x_star, x, c, l, sigma, v, kappa)
        mu_star = Kuz @ alpha
        
        # Check for NaN/inf in predictions
        if not np.all(np.isfinite(mu_star)):
            print(f"  [Sample {i}] Warning: NaN/inf in mu_star, skipping")
            pred_means[i, :] = 0.0
        else:
            pred_means[i, :] = mu_star

        # Predictive variance (diagonal)
        try:
            Kxx_inv_KuzT = solve(Kxx, Kuz.T)
        except np.linalg.LinAlgError:
            # Add extra jitter and try again
            Kxx += 1e-6 * np.eye(Kxx.shape[0])
            Kxx_inv_KuzT = solve(Kxx, Kuz.T)
        Kuu = rbf_kernel(x_star, x_star, l, sigma)
        pred_cov_diag = np.diag(Kuu) - np.sum(Kuz * (Kxx_inv_KuzT.T), axis=1)
        pred_cov_diag = np.maximum(pred_cov_diag, 0.0)
        
        if not np.all(np.isfinite(pred_cov_diag)):
            print(f"  [Sample {i}] Warning: NaN/inf in pred_cov_diag")
            pred_vars[i, :] = 1e-3
        else:
            pred_vars[i, :] = pred_cov_diag

    # Combine samples using law of total variance
    post_mean = np.mean(pred_means, axis=0)
    E_var = np.mean(pred_vars, axis=0)
    E_mean_sq = np.mean(pred_means**2, axis=0)
    Var_mean = E_mean_sq - post_mean**2
    post_var = np.maximum(E_var + Var_mean, 0.0)  # Ensure non-negative variance

    return {
        "post_mean": post_mean,
        "post_var": post_var,
        "elbo_trace": elbo_vals,
        "final_mu": var_dist.mu,
        "final_logsigma": var_dist.logsigma
    }

# ---------------------- main script ----------------------
# Data generation
a, b = 0.0, 5.0
n = 100
m = 70
nt = 70
nc = 10
x = np.sort(np.random.uniform(a, b, n-2))
c = np.sort(np.random.uniform(a, b, nc))
u = np.linspace(a, b, m)
x_star = np.sort(np.random.uniform(a, b, 30))
idx_u = np.arange(0, m)
kappa_true = 0.6
v_true = 0.3

yt = u_exact(x_star, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x_star))
yu = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))

yb = np.array([0.0, 0.0])
x = np.hstack([a, b, x, c])
yf = f(c) + np.random.normal(0, 0.1, size=len(c))
y = np.hstack([yb, yu, yf])
# Initial guess for hyperparameters (same as before)
init_theta = [0.2, 0.3, 1e-2, 0.8, 0.2]

# Run variational inference
vi_out = variational_inference(x, c, y, x_star,
                               init_theta=init_theta,
                               with_bc=True,
                               lr=0.01,
                               lr_decay=0.995,
                               n_iter=100,
                               batch_size=50,
                               n_pred_samples=500)

print("Posterior mean shape:", vi_out["post_mean"].shape)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_star, vi_out["post_mean"], label="Posterior Mean")
plt.fill_between(x_star,
                 vi_out["post_mean"] - 2*np.sqrt(vi_out["post_var"]),
                 vi_out["post_mean"] + 2*np.sqrt(vi_out["post_var"]),
                 color='lightblue', alpha=0.5, label="Posterior ±2 std")
plt.scatter(x, y[:len(x)], color='red', label="Observations", zorder=1)
plt.scatter(x_star, yt, color='green', label="Test points (x_star)", zorder=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Variational Bayesian Hyperparameter Inference")
plt.legend()
plt.show()

# Plot ELBO convergence
plt.figure(figsize=(8, 4))
plt.plot(vi_out["elbo_trace"])
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO during optimisation")
plt.grid(True)
plt.show()
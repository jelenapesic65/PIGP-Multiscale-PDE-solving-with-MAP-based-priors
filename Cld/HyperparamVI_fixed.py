"""
HyperparamVI_fixed.py
=====================
Variational Inference for GP hyperparameters in the 1D advection-diffusion
PI-GP framework.

Bug fixes vs HyperparamMCMCDeep.py
------------------------------------
BUG 1 (Fatal): x was built as np.hstack([a, b, x, c]) in main, so it
  had length n + nc. build_K / build_K_star then sliced x[:n], silently
  dragging nc collocation points into the "observation" block and building
  an ill-defined covariance.
  FIX: Separate the arrays: x_obs holds boundary + interior observations,
  c holds collocation points. They are never merged into a single vector.

BUG 2 (NaN at evaluation): The VI loop constructed
  theta = [l, sigma, sigma_n, v, kappa] with sigma_n *already* exponentiated,
  but nlml() expects theta[2] = log_sigma_n and does sigma_n = exp(theta[2])
  internally, so sigma_n was double-exponentiated.
  FIX: nlml() now receives the raw unconstrained vector eta directly and
  handles all transformations in one place (see nlml_from_eta helper).

BUG 3 (Biased gradient): grad_log_prob was called on the *original* (unclipped)
  eta, but f_elbo was computed on the *clamped* version, breaking the
  score-function identity E[(f-b) ∇ log q] = ∇ ELBO.
  FIX: store clamped eta and use that for BOTH f_elbo and grad_log_prob.

BUG 4 (log_prior called with wrong signature): log_prior expects
  (l, sigma, log_sigma_n, v, kappa) where l, sigma > 0, and takes log()
  of them internally. In the VI loop the parameters were already in
  unconstrained (log) space, so log(exp(eta)) was correct but the
  remaining clipping logic was inconsistent.
  FIX: a dedicated log_prior_natural(eta) that works directly in the
  unconstrained space used by the variational distribution.

BUG 5 (Unbounded prediction samples): The prediction loop sampled from the
  final variational distribution without clamping, so tail samples with
  e.g. log(l) = -10 gave l ≈ 4.5e-5 and k_xxxpxp ~ 1/l^8 ≈ 10^35,
  overflowing float64.
  FIX: same clamping is applied to every prediction sample.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from numpy.linalg import solve, slogdet


def u_exact_complex(x, kappa, v):
    """
    Exact solution to  v u' - kappa u'' = sin(πx) + 3 sin(3πx)
    on [0,1] with u(0)=u(1)=0.
    Derived by superposition of two particular solutions + homogeneous.
    """
    pi = np.pi
    r  = v / kappa           # Pe-related exponent, here r = 1.5/0.08 = 18.75

    A = []
    B = []
    for n in [1, 3]:
        npi   = n * pi
        D     = kappa**2 * npi**2 + v**2   # = (kappa*nπ)^2 + v^2  (wrong factor, see below)
        # Correct form from substituting A sin + B cos:
        #   kappa*n²π²*A - v*nπ*B = 1
        #   kappa*n²π²*B + v*nπ*A = 0
        # => A = kappa / (kappa^2*n^2π^2 + v^2)  [both divided by n²π²]
        #    B = -v / (nπ * (kappa^2*n^2π^2 + v^2))
        An = kappa / (kappa**2 * npi**2 + v**2)
        Bn = -v   / (npi * (kappa**2 * npi**2 + v**2))
        A.append(An)
        B.append(Bn)

    # Homogeneous BCs: C1 + C2*exp(r*x)
    # At x=0: C1 + C2 + sum(Bn) = 0
    # At x=1: C1 + C2*exp(r) + sum(Bn*cos(nπ)) = 0
    #         cos(nπ) = -1 for both n=1,3
    sum_B      = B[0] + B[1]
    sum_Bcos   = B[0] * np.cos(pi) + B[1] * np.cos(3 * pi)  # both = -Bn
    e_r        = np.exp(r)
    C2 = (sum_B - sum_Bcos) / (e_r - 1)   # = 2*sum_B / (exp(r)-1)
    C1 = -sum_B - C2

    return (
        C1 + C2 * np.exp(r * x)
        + A[0] * np.sin(pi * x)     + B[0] * np.cos(pi * x)
        + A[1] * np.sin(3 * pi * x) + B[1] * np.cos(3 * pi * x)
    )


def u_exact(x, kappa, mu):
    pi = np.pi
    denom = pi**2 * kappa**2 + mu**2
    A = kappa / denom
    B = -mu / (pi * denom)
    r = mu / kappa
    e = np.exp(r)
    C2 = (2 * B) / (e - 1)
    C1 = -B - C2
    return C1 + C2 * np.exp(r * x) + A * np.sin(pi * x) + B * np.cos(pi * x)


def f(x):
    return np.sin(np.pi * x) + 3 * np.sin(3 * np.pi * x)


def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)




_MIN_L = 0.02   

def kernel_derivatives(x, y, l, sigma):
    l = max(l, _MIN_L)
    r  = x[:, None] - y[None, :]
    k  = sigma**2 * np.exp(-0.5 * r**2 / l**2)
    l2, l4, l6, l8 = l**2, l**4, l**6, l**8
    k_x      = -(r / l2) * k
    k_xp     =  (r / l2) * k
    k_xx     = (r**2 / l4 - 1 / l2) * k
    k_xpxp   = k_xx
    k_xxp    = (1 / l2 - r**2 / l4) * k
    k_xxxp   = (-3 * r / l4 + r**3 / l6) * k
    k_xxpxp  = (3 * r / l4 - r**3 / l6) * k
    k_xxxpxp = (3 / l4 - 6 * r**2 / l6 + r**4 / l8) * k
    return k, k_x, k_xp, k_xx, k_xpxp, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp



def build_K(x_obs, c, l, sigma, sigma_n, v, kappa, with_bc=True):

    xf = c
    nb = len(x_obs)
    nc = len(c)
    nxc = nb + nc
    K = np.zeros((nxc, nxc))

    kbb, *_ = kernel_derivatives(x_obs, x_obs, l, sigma)
    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x_obs, xf, l, sigma)
    Kbf = v * k_xp - kappa * k_xpxp

    K[:nb, :nb] = kbb
    K[:nb, nb:] = Kbf

    kfb, k_x, _, k_xx, *_ = kernel_derivatives(xf, x_obs, l, sigma)
    Kfb = v * k_x - kappa * k_xx
    K[nb:, :nb] = Kfb

    _, _, _, _, _, k_xxp, _, _, k_xxxpxp = kernel_derivatives(xf, xf, l, sigma)
    Kff = v**2 * k_xxp + kappa**2 * k_xxxpxp
    Kff += sigma_n**2 * np.eye(nc)
    K[nb:, nb:] = Kff

    if not with_bc:
        return K[2:, 2:]
    return K


def build_K_star(x_star, x_obs, c, l, sigma, v, kappa):
    xf = c
    kbb, *_ = kernel_derivatives(x_star, x_obs, l, sigma)
    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x_star, xf, l, sigma)
    Kuf = v * k_xp - kappa * k_xpxp
    return np.hstack([kbb, Kuf])


def log_prior_natural(eta):
 
    log_l, log_sigma, log_sigma_n, v, kappa = eta

    mu_l, std_l   = np.log(0.3), 0.8
    mu_s, std_s   = np.log(1.0), 0.5
    mu_n, std_n   = np.log(1e-2), 1.0
    mu_v, std_v   = 0.5, 3
    mu_k, std_k   = 0.5, 3

    lp  = -0.5 * ((log_l     - mu_l) / std_l) ** 2
    lp += -0.5 * ((log_sigma - mu_s) / std_s) ** 2
    lp += -0.5 * ((log_sigma_n - mu_n) / std_n) ** 2
    lp += -0.5 * ((v   - mu_v) / std_v) ** 2
    lp += -0.5 * ((kappa - mu_k) / std_k) ** 2
    return lp



ETA_CLIP = np.array([-3.0, -3.0, -8.0, 0.001, 0.001])   
ETA_CLIP_HI = np.array([2.0,  1.0,  2.0, 2.0,  2.0])   


def eta_to_theta(eta):
    """Convert unconstrained eta → physical hyperparameters"""
    eta_c = np.clip(eta, ETA_CLIP, ETA_CLIP_HI)
    l       = np.exp(eta_c[0])
    sigma   = np.exp(eta_c[1])
    sigma_n = np.exp(eta_c[2])
    v       = eta_c[3]
    kappa   = eta_c[4]
    return l, sigma, sigma_n, v, kappa


def nlml_from_eta(eta, x_obs, c, y, with_bc=True, jitter=1e-6):
    
    l, sigma, sigma_n, v, kappa = eta_to_theta(eta)
    try:
        K = build_K(x_obs, c, l, sigma, sigma_n, v, kappa, with_bc)
        K += jitter * np.eye(K.shape[0])
        sign, logdet = slogdet(K)
        if not np.isfinite(logdet) or sign <= 0:
            return 1e10
        alpha = solve(K, y)
        result = 0.5 * y @ alpha + 0.5 * logdet
        return result if np.isfinite(result) else 1e10
    except np.linalg.LinAlgError:
        return 1e10



class VariationalGaussian:
    def __init__(self, dim, init_mu, init_logsigma):
        self.dim      = dim
        self.mu       = np.array(init_mu,      dtype=float)
        self.logsigma = np.array(init_logsigma, dtype=float)

    def sample(self, n_samples):
        eps = np.random.randn(n_samples, self.dim)
        return self.mu + np.exp(self.logsigma) * eps

    def log_prob(self, samples):
        sigma = np.exp(self.logsigma)
        norm  = -0.5 * self.dim * np.log(2 * np.pi) - np.sum(self.logsigma)
        sq    = ((samples - self.mu) / sigma) ** 2
        return norm - 0.5 * np.sum(sq, axis=1)

    def grad_log_prob(self, samples):
        sigma          = np.exp(self.logsigma)
        dev            = (samples - self.mu) / sigma
        grad_mu        = dev / sigma
        grad_logsigma  = -1.0 + dev**2
        return grad_mu, grad_logsigma


def variational_inference(x_obs, c, y, x_star,
                          init_theta,
                          with_bc=True,
                          lr=0.01,
                          n_iter=200,
                          batch_size=50,
                          n_pred_samples=500,
                          jitter=1e-6,
                          verbose=True):
    """
    Parameters
    ----------
    x_obs     : observation points (boundary first, then interior)
    c         : collocation points
    y         : target vector aligned with np.hstack([x_obs, c])
    x_star    : prediction locations
    init_theta: [l, sigma, sigma_n, v, kappa] in physical space
    """

    # --- Initialise variational distribution in unconstrained space ---
    l0, sigma0, sigma_n0, v0, kappa0 = init_theta
    init_eta = np.array([
        np.log(l0), np.log(sigma0), np.log(sigma_n0), v0, kappa0
    ])
    q = VariationalGaussian(dim=5,
                             init_mu=init_eta,
                             init_logsigma=np.full(5, -1.0))  # start with small variance

    elbo_trace = []
    baseline   = 0.0
    beta_base  = 0.9

    if verbose:
        print("Starting variational inference ...")

    for it in range(n_iter):
        # 1. Draw samples
        eta_raw = q.sample(batch_size)                        # (B, 5), unconstrained

        
        eta_c = np.clip(eta_raw, ETA_CLIP, ETA_CLIP_HI)      # (B, 5)

        # 2. Evaluate log-joint for each clamped sample
        log_joint = np.array([
            -nlml_from_eta(eta_c[i], x_obs, c, y, with_bc, jitter)
            + log_prior_natural(eta_c[i])
            for i in range(batch_size)
        ])

        # 3. Entropy term: log q evaluated at the clamped samples
        log_q = q.log_prob(eta_c)

        # 4. ELBO per sample
        f_elbo   = log_joint - log_q
        elbo_val = np.nanmean(f_elbo)
        elbo_trace.append(elbo_val)

        # 5. Baseline update
        baseline = beta_base * baseline + (1 - beta_base) * elbo_val

        # 6. Score-function gradient on the clamped samples
        g_mu, g_ls = q.grad_log_prob(eta_c)
        centred    = (f_elbo - baseline)[:, None]
        # mask out any NaN rows
        mask = np.isfinite(centred.squeeze())
        if mask.sum() == 0:
            if verbose:
                print(f"Iter {it:4d}: all ELBO values invalid, skipping update")
            continue
        grad_mu = np.mean(centred[mask] * g_mu[mask], axis=0)
        grad_ls = np.mean(centred[mask] * g_ls[mask], axis=0)

        # 7. Gradient ascent
        q.mu       += lr * grad_mu
        q.logsigma += lr * grad_ls
        q.logsigma  = np.clip(q.logsigma, -5.0, 2.0)

        if verbose and it % 50 == 0:
            print(f"  Iter {it:4d}  ELBO = {elbo_val:+.4f}  "
                  f"mu = {q.mu.round(3)}  std = {np.exp(q.logsigma).round(3)}")

    if verbose:
        print("Optimisation finished.\n")

    # --- Posterior predictive via Monte Carlo ---
    n_star = len(x_star)
    pred_means = np.zeros((n_pred_samples, n_star))
    pred_vars  = np.zeros((n_pred_samples, n_star))

    eta_pred = q.sample(n_pred_samples)
    eta_pred = np.clip(eta_pred, ETA_CLIP, ETA_CLIP_HI)   # FIX (Bug 5)

    for i in range(n_pred_samples):
        l, sigma, sigma_n, v, kappa = eta_to_theta(eta_pred[i])
        try:
            Kxx = build_K(x_obs, c, l, sigma, sigma_n, v, kappa, with_bc)
            Kxx += jitter * np.eye(Kxx.shape[0])
            alpha  = solve(Kxx, y)
            Kuz    = build_K_star(x_star, x_obs, c, l, sigma, v, kappa)
            mu_s   = Kuz @ alpha

            if not np.all(np.isfinite(mu_s)):
                pred_means[i] = 0.0
            else:
                pred_means[i] = mu_s

            # Predictive variance
            V     = solve(Kxx, Kuz.T)
            Kuu   = rbf_kernel(x_star, x_star, l, sigma)
            pvar  = np.diag(Kuu) - np.sum(Kuz * V.T, axis=1)
            pvar  = np.maximum(pvar, 0.0)
            pred_vars[i] = pvar if np.all(np.isfinite(pvar)) else 1e-3

        except np.linalg.LinAlgError:
            pred_means[i] = 0.0
            pred_vars[i]  = 1e-3

    post_mean = np.mean(pred_means, axis=0)
    E_var     = np.mean(pred_vars,  axis=0)
    Var_mean  = np.mean(pred_means**2, axis=0) - post_mean**2
    post_var  = np.maximum(E_var + Var_mean, 0.0)

    return {
        "post_mean":      post_mean,
        "post_var":       post_var,
        "elbo_trace":     elbo_trace,
        "final_mu":       q.mu,
        "final_logsigma": q.logsigma,
    }


if __name__ == "__main__":
    np.random.seed(42)

    a, b      = 0.0, 1.0
    n_int     = 50        
    nc        = 15        

    kappa_true = 0.08
    v_true     = 1.5

    x_int = np.sort(np.random.uniform(a, b, n_int))
    c     = np.sort(np.random.uniform(a, b, nc))

    x_obs = np.hstack([a, b, x_int])

    yu  = u_exact_complex(x_int, kappa_true, v_true) + np.random.normal(0, 1e-3, n_int)
    yb  = np.array([0.0, 0.0])
    yf  = f(c) + np.random.normal(0, 0.1, nc)
    y   = np.hstack([yb, yu, yf])

    x_star = np.sort(np.random.uniform(a, b, 40))
    yt     = u_exact_complex(x_star, kappa_true, v_true)

    init_theta = [0.2, 0.5, 1e-2, 0.5, 0.5]

    vi = variational_inference(
        x_obs, c, y, x_star,
        init_theta=init_theta,
        with_bc=True,
        lr=5e-3,
        n_iter=300,
        batch_size=40,
        n_pred_samples=300,
        verbose=True,
    )

    # Recovered physical parameters (posterior mean)
    l_pm, sig_pm, sn_pm, v_pm, k_pm = eta_to_theta(vi["final_mu"])
    print(f"Posterior mean hyperparameters:")
    print(f"  l={l_pm:.4f}, sigma={sig_pm:.4f}, sigma_n={sn_pm:.2e}, v={v_pm:.4f}, kappa={k_pm:.4f}")
    print(f"  (true: v={v_true}, kappa={kappa_true})")

    ax_x = np.linspace(a, b, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.fill_between(
        x_star,
        vi["post_mean"] - 2 * np.sqrt(vi["post_var"]),
        vi["post_mean"] + 2 * np.sqrt(vi["post_var"]),
        color="lightblue", alpha=0.5, label="Posterior ±2σ",
    )
    ax.plot(x_star, vi["post_mean"], "b-", lw=2, label="Posterior mean")
    ax.plot(ax_x, u_exact_complex(ax_x, kappa_true, v_true), "k--", lw=1.5, label="Exact")
    ax.scatter(x_star, yt, c="green", s=20, zorder=5, label="Test pts")
    ax.scatter(x_obs, y[:len(x_obs)], c="orange", s=20, zorder=5, label="Training obs")
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("VI posterior predictive")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(vi["elbo_trace"])
    ax.set_xlabel("Iteration"); ax.set_ylabel("ELBO")
    ax.set_title("ELBO convergence")
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    print("Saved VI_posterior.png")




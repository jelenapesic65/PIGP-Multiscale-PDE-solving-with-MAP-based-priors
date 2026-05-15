"""

PDE:  mu u_x = kappa u_xx + f(x),   u(0)=u(1)=0  

1. Choose PDE params mu, kappa.
2. Set prior means on the forcing function:
       l_f     = true forcing lengthscale
       sigma_f = sqrt(sigma_scale * l_f^2 / kappa)   [paper-motivated scaling]
3. Draw f ~ GP(0, k_{l_f, sigma_f}) and solve PDE for u.
4. Add observation noise -> u_obs
5. Infer (l_f, sigma_f) via
       a) MCMC full posterior 
       b) ML-II with Gaussian prior as regulariser (MAP)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import truncnorm
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

rng = np.random.default_rng(42)

# 1.
mu     = 0.5     # advection speed
kappa = 0.3     # diffusion coefficient

l_f         = 0.4   # true forcing lengthscale (controls smoothness of f and u)
sigma_scale = 1.0   # true forcing variance scale
sigma_f     = np.sqrt(sigma_scale * l_f**2 / kappa)   # prior mean amplitude


#this is yet to be determined!!!!!!!
prior_std_l     = 0.15
prior_std_sigma = 0.20

print(f"PDE:  mu={mu},  kappa={kappa}")
print(f"True forcing GP:  l_f={l_f:.3f},  sigma_f={sigma_f:.4f}")
print(f"  (sigma_f^2 = {sigma_f**2:.4f} = l_f^2/kappa = {l_f**2/kappa:.4f})")
print(f"Priors:  l ~ N({l_f}, {prior_std_l}^2),  sigma ~ N({sigma_f:.3f}, {prior_std_sigma}^2)")

# 2. 
N     = 60                                 # number of interior points
x_int = np.linspace(0, 1, N + 2)[1:-1]    # interior x-values
X     = x_int[:, None]                     # (N,1) for kernel calls
dx    = x_int[1] - x_int[0]
noise_std = 0.05

def se_kernel(X1, X2, sigma, l):
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

# Sample forcing function
K_f    = se_kernel(X, X, sigma_f, l_f) + 1e-8 * np.eye(N)
f_true = np.linalg.cholesky(K_f) @ rng.standard_normal(N)

# 3.  PDE operator - upwind first-derivative + central second-derivative to find true solution

# Upwind first-derivative (no wrap — boundary rows vanish)
D1 = (np.diag(np.ones(N)) - np.diag(np.ones(N - 1), -1)) / dx

# Central second-derivative (Dirichlet: boundary values = 0)
D2 = (np.diag(-2 * np.ones(N))
      + np.diag(np.ones(N - 1),  1)
      + np.diag(np.ones(N - 1), -1)) / dx**2

A_pde  = kappa * D2 - mu * D1      
u_true = np.linalg.solve(A_pde, -f_true)
u_obs  = u_true + noise_std * rng.standard_normal(N)
A_inv  = np.linalg.inv(A_pde)

print(f"\nPDE solved:  u in [{u_true.min():.4f}, {u_true.max():.4f}]")
print(f"cond(A) = {np.linalg.cond(A_pde):.1f}")

# 4.  GP marginal likelihood 
#     Propagated covariance:  K_uu = A_inv K_ff A_inv^T + sigma_n^2 I


#Small comment: Kuu in general wouldn't be built like this, since in general I do not want to verify the priors on the f
#Since now I build the forcing function based on known parameters I want to use this as verification of the significance of priors
#Usually I ask "what GP prior on u best explains u_obs?" rather than "what GP prior on f best explains u_obs after propagating through the PDE?".


def nlml(l, sigma):
    """Negative log marginal likelihood (exact)."""
    K_ff = se_kernel(X, X, sigma, l) + 1e-8 * np.eye(N) #Do I want to use this trick? 
    K_uu = A_inv @ K_ff @ A_inv.T + noise_std**2 * np.eye(N) # u = A_inv f; u ~ N(0, A_inv K_ff A_inv^T + sigma_n^2 I)
    
    #K_uu = se_kernel(X, X, sigma, l) + noise_std**2 * np.eye(N)
    
    try:
        c, low = cho_factor(K_uu)
        alpha  = cho_solve((c, low), u_obs)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        return 0.5 * (u_obs @ alpha + logdet + N * np.log(2 * np.pi))
    except Exception:
        return np.inf

print(f"nlml at true hyperparams: {nlml(l_f, sigma_f):.3f}")

# 5.  MCMC via PyMC  (Metropolis)
class LogLikOp(Op):
    #Wraps nlml as a gradient-free PyTensor Op for use as a PyMC Potential
    def make_node(self, l, sigma):
        l     = pt.as_tensor_variable(l)
        sigma = pt.as_tensor_variable(sigma)
        return Apply(self, [l, sigma],
                     [pt.TensorType("float64", shape=())()])
    #define input output
    def perform(self, node, inputs, outputs):
        val = -nlml(float(inputs[0]), float(inputs[1]))
        outputs[0][0] = np.array(val, dtype=np.float64)
    #no gradient info, return zeros
    def grad(self, inputs, g):
        return [pt.zeros_like(inputs[0]), pt.zeros_like(inputs[1])]

#init object
loglik_op = LogLikOp()

print("\n── Running MC (Metropolis) ──")
with pm.Model() as MCMC_model:
    l_rv     = pm.TruncatedNormal("l",
                                  mu=l_f,     sigma=prior_std_l,
                                  lower=0.05, upper=1.5)
    sigma_rv = pm.TruncatedNormal("sigma",
                                  mu=sigma_f,  sigma=prior_std_sigma,
                                  lower=0.05, upper=3.0)
    pm.Potential("ll", loglik_op(l_rv, sigma_rv))

    trace = pm.sample(
        draws=1500, tune=800, chains=1,
        cores=1,
        step=pm.Metropolis(),
        progressbar=True,
        random_seed=0,
    )

l_samp     = trace.posterior["l"].values.flatten()
sigma_samp = trace.posterior["sigma"].values.flatten()
print(f"Posterior  l:      {l_samp.mean():.3f} +/- {l_samp.std():.3f}  (true {l_f})")
print(f"Posterior  sigma:  {sigma_samp.mean():.3f} +/- {sigma_samp.std():.3f}  (true {sigma_f:.3f})")

# 
# 6.  ML-II with Gaussian prior as regulariser  (MAP via Nelder-Mead)

def neg_log_posterior(params):
    l_v, s_v = params
    if l_v <= 0 or s_v <= 0:
        return 1e10
    return (nlml(l_v, s_v)
            + 0.5 * ((l_v - l_f)     / prior_std_l)**2
            + 0.5 * ((s_v - sigma_f) / prior_std_sigma)**2)

best_res, best_val = None, np.inf
for l0, s0 in [(l_f, sigma_f),
               (l_f * 0.6, sigma_f * 0.8),
               (l_f * 1.4, sigma_f * 1.3),
               (0.25, 0.35), (0.55, 0.9)]:
    res = minimize(neg_log_posterior, [l0, s0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    if res.fun < best_val and res.success:
        best_val, best_res = res.fun, res

if best_res is None:  
    best_res = min(
        [minimize(neg_log_posterior, [l0, s0], method="Nelder-Mead")
         for l0, s0 in [(l_f, sigma_f), (0.3, 0.6), (0.5, 0.8)]],
        key=lambda r: r.fun
    )

l_map, sigma_map = best_res.x
print(f"\nML-II MAP (prior reg.):  l={l_map:.4f},  sigma={sigma_map:.4f}  "
      f"(true: l={l_f}, sigma={sigma_f:.4f})")

# ── Pure ML-II: minimise NLML only, no prior term ────────────────────────
def pure_nlml(params):
    l_v, s_v = params
    if l_v <= 0 or s_v <= 0:
        return 1e10
    return nlml(l_v, s_v)

best_res2, best_val2 = None, np.inf
for l0, s0 in [(l_f, sigma_f),
               (l_f * 0.6, sigma_f * 0.8),
               (l_f * 1.4, sigma_f * 1.3),
               (0.25, 0.35), (0.55, 0.9)]:
    res2 = minimize(pure_nlml, [l0, s0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    if res2.fun < best_val2:
        best_val2, best_res2 = res2.fun, res2

l_mle, sigma_mle = best_res2.x
print(f"ML-II (no prior):        l={l_mle:.4f},  sigma={sigma_mle:.4f}  "
      f"(true: l={l_f}, sigma={sigma_f:.4f})")


fig = plt.figure(figsize=(15, 7))
gs  = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.40)
x1d = x_int

# Prior PDFs for overlay
l_grid = np.linspace(0.02, 1.2, 400)
s_grid = np.linspace(0.02, 1.5, 400)
al  = (0.05 - l_f)     / prior_std_l;      bl  = (1.5 - l_f)     / prior_std_l
as_ = (0.05 - sigma_f) / prior_std_sigma;  bs  = (3.0 - sigma_f) / prior_std_sigma
prior_l_pdf = truncnorm.pdf(l_grid, al,  bl,  loc=l_f,     scale=prior_std_l)
prior_s_pdf = truncnorm.pdf(s_grid, as_, bs,  loc=sigma_f, scale=prior_std_sigma)

# ── (0,0)  Forcing sample + PDE solution ────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(x1d, f_true, color="steelblue",  lw=1.8, label=r"$f(x)$ — GP sample")
ax.plot(x1d, u_true, color="tomato",     lw=1.5, ls="--", label=r"$u_\mathrm{true}$")
ax.plot(x1d, u_obs,  "k.", ms=3.5, alpha=0.55,   label=r"$u_\mathrm{obs}$ (noisy)")
ax.set_xlabel("x"); ax.set_ylabel("value")
ax.set_title(
    f"Forcing sample & PDE solution\n"
    r"$\mu$=" + f"{mu},  " + r"$\kappa$=" + f"{kappa},  "
    f"$l_f$={l_f},  $\\sigma$={sigma_f:.2f}"
)
ax.legend(fontsize=7.5, loc="upper right")

# ── (0,1)  MCMC lengthscale histogram ────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.hist(l_samp, bins=40, color="steelblue", alpha=0.70, density=True,
        label="MCMC posterior")
ax.plot(l_grid, prior_l_pdf, color="grey", lw=1.8, ls=":", label="Prior")
ax.axvline(l_f,           color="black",  lw=2.2, ls="--",
           label=f"True $\\ell$ = {l_f}")
ax.axvline(l_samp.mean(), color="tomato", lw=2.2, ls="-",
           label=f"Post. mean = {l_samp.mean():.3f}")
ax.set_xlabel(r"Lengthscale $\ell$"); ax.set_ylabel("Density")
ax.set_title("MCMC — Lengthscale identification")
ax.legend(fontsize=7.5)

# ── (0,2)  MCMC amplitude histogram ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.hist(sigma_samp, bins=40, color="mediumseagreen", alpha=0.70, density=True,
        label="MCMC posterior")
ax.plot(s_grid, prior_s_pdf, color="grey", lw=1.8, ls=":", label="Prior")
ax.axvline(sigma_f,           color="black",  lw=2.2, ls="--",
           label=f"True $\\sigma$ = {sigma_f:.3f}")
ax.axvline(sigma_samp.mean(), color="tomato", lw=2.2, ls="-",
           label=f"Post. mean = {sigma_samp.mean():.3f}")
ax.set_xlabel(r"Amplitude $\sigma$"); ax.set_ylabel("Density")
ax.set_title("MCMC — Amplitude identification")
ax.legend(fontsize=7.5)

# ── (1,0)  NLML surface + MCMC samples + MAP ─────────────────────────────
print("\nComputing NLML surface for plot (60x60 grid)...")
ax = fig.add_subplot(gs[1, 0])
lg_  = np.linspace(0.05, 1.0, 60)
sg_  = np.linspace(0.05, 1.8, 60)
ZZ   = np.array([[nlml(ll, ss) for ll in lg_] for ss in sg_])
ZZ_c = np.clip(ZZ, np.nanpercentile(ZZ, 3), np.nanpercentile(ZZ, 97))
cf = ax.contourf(lg_, sg_, ZZ_c, levels=22, cmap="viridis")
ax.contour(lg_, sg_, ZZ_c, levels=22, colors="black", linewidths=0.3, alpha=0.4)
ax.plot(l_f,   sigma_f,   "r+", ms=14, mew=1.5, zorder=1, label="True")
ax.plot(l_map, sigma_map, "w*", ms=11, mew=1.5, zorder=1, label="ML-II MAP")
ax.scatter(l_samp[::4], sigma_samp[::4],
           c="white", s=5, alpha=0.30, zorder=1, label="MCMC samples")
plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\sigma$")
ax.set_title("NLML surface  (MCMC samples + MAP)")
ax.legend(fontsize=7.5, loc="upper right")

# ── (1,1)  ML-II vs MCMC — lengthscale ───────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.hist(l_samp, bins=40, color="steelblue", alpha=0.50, density=True,
        label="MCMC posterior")
ax.plot(l_grid, prior_l_pdf, color="grey", lw=1.8, ls=":", label="Prior")
ax.axvline(l_f,   color="black",      lw=2.2, ls="--",
           label=f"True $\\ell$ = {l_f}")
ax.axvline(l_map, color="darkorange", lw=1.5, ls="-",
           label=f"ML-II MAP = {l_map:.3f}")
ax.axvline(l_samp.mean(), color="tomato", lw=2.2, ls="-",
           label=f"MCMC mean = {l_samp.mean():.3f}")
ax.set_xlabel(r"Lengthscale $\ell$"); ax.set_ylabel("Density")
ax.set_title("ML-II (prior reg.) vs MCMC\nLengthscale")
ax.legend(fontsize=7.5)

# ── (1,2)  ML-II vs MCMC — amplitude ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.hist(sigma_samp, bins=40, color="mediumseagreen", alpha=0.50, density=True,
        label="MCMC posterior")
ax.plot(s_grid, prior_s_pdf, color="grey", lw=1.8, ls=":", label="Prior")
ax.axvline(sigma_f,   color="black",      lw=2.2, ls="--",
           label=f"True $\\sigma$ = {sigma_f:.3f}")
ax.axvline(sigma_map, color="darkorange", lw=1.5, ls="-",
           label=f"ML-II MAP = {sigma_map:.3f}")
ax.axvline(sigma_samp.mean(), color="tomato", lw=2.2, ls="-",
           label=f"MCMC mean = {sigma_samp.mean():.3f}")
ax.set_xlabel(r"Amplitude $\sigma$"); ax.set_ylabel("Density")
ax.set_title("ML-II (prior reg.) vs MCMC\nAmplitude")
ax.legend(fontsize=7.5)

fig.suptitle(
    "Advection-Diffusion: GP forcing-function hyperparameter inference",
    fontsize=13, y=1.01
)
plt.tight_layout()
out = "advdiff_inference.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved {out}")

fig2 = plt.figure(figsize=(6, 12))
gs2  = gridspec.GridSpec(3, 1, hspace=0.40, wspace=0.30)

# ── (0,0)  NLML surface with pure ML-II ─────────────────────────────────
ax = fig2.add_subplot(gs2[0, 0])
cf2 = ax.contourf(lg_, sg_, ZZ_c, levels=22, cmap="viridis")
ax.contour(lg_, sg_, ZZ_c, levels=22, colors="black", linewidths=0.3, alpha=0.4)
ax.plot(l_f,    sigma_f,   "r+", ms=10, mew=1.5, zorder=1, label="True")
ax.plot(l_mle,  sigma_mle, "ws", ms=5, mew=1.5, zorder=1, label="ML-II (no prior)")
ax.plot(l_map,  sigma_map, "w*", ms=8, mew=1.5, zorder=1, label="ML-II (prior)")
ax.scatter(l_samp[::4], sigma_samp[::4],
           c="white", s=5, alpha=0.30, zorder=1, label="MCMC samples")
plt.colorbar(cf2, ax=ax, fraction=0.046, pad=0.04)
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\sigma$")
ax.set_title("NLML surface")
ax.legend(fontsize=7.5, loc="upper right")

# ── (1,0)  No-prior ML-II vs prior ML-II vs MCMC — lengthscale ───────────
ax = fig2.add_subplot(gs2[1, 0])
ax.hist(l_samp, bins=40, color="steelblue", alpha=0.50, density=True,
        label="MCMC posterior")
ax.plot(l_grid, prior_l_pdf, color="grey", lw=1.8, ls=":", label="Prior")
ax.axvline(l_f,    color="black",      lw=2.2, ls="--",
           label=f"True $\\ell$ = {l_f}")
ax.axvline(l_map,  color="darkorange", lw=1.5, ls="-",
           label=f"ML-II+prior = {l_map:.3f}")
ax.axvline(l_mle,  color="mediumpurple", lw=1.5, ls="-.",
           label=f"ML-II (no prior) = {l_mle:.3f}")
ax.axvline(l_samp.mean(), color="tomato", lw=2.2, ls="-",
           label=f"MCMC mean = {l_samp.mean():.3f}")
ax.set_xlabel(r"$l$"); ax.set_ylabel("Density")
ax.set_title("Lengthscale", fontsize=12)
ax.legend(fontsize=7.5)

# ── (2,0)  No-prior ML-II vs prior ML-II vs MCMC — amplitude ─────────────
ax = fig2.add_subplot(gs2[2, 0])
ax.hist(sigma_samp, bins=40, color="mediumseagreen", alpha=0.50, density=True,
        label="MCMC posterior")
ax.plot(s_grid, prior_s_pdf, color="grey", lw=1.8, ls=":", label="Prior")
ax.axvline(sigma_f,    color="black",       lw=2.2, ls="--",
           label=f"True $\\sigma$ = {sigma_f:.3f}")
ax.axvline(sigma_map,  color="darkorange",  lw=1.5, ls="-",
           label=f"ML-II+prior = {sigma_map:.3f}")
ax.axvline(sigma_mle,  color="mediumpurple", lw=1.5, ls="-.",
           label=f"ML-II (no prior) = {sigma_mle:.3f}")
ax.axvline(sigma_samp.mean(), color="tomato", lw=2.2, ls="-",
           label=f"MCMC mean = {sigma_samp.mean():.3f}")
ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("Density")
ax.set_title("Amplitude", fontsize=12)
ax.legend(fontsize=7.5)


plt.tight_layout()
out2 = "advdiff_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out2}")
"""

=============================
Compares classic GPR vs PI-GP (GPR with physics-informed kernel and
physics-derived kernel / collocation points) under three prior regimes:

  1. PHYSICS_DERIVED  – priors from energy arguments and Pe-number bounds
  2. NON_INFORMATIVE  – weakly informative / vague Gaussians
  3. MISSPECIFIED     – priors centred far from the true values

For each (method × prior) combination the script reports:
  - MAP hyperparameter estimates
  - RMSE at test locations
  - Time to optimise
  - NLML at MAP

Run:  python compare_priors_gpr_kissgp.py

Requires: numpy, scipy, matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, slogdet
from scipy.optimize import minimize

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


def f(x):
    return np.sin(np.pi * x) + 3 * np.sin(3 * np.pi * x)

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


#def f(x):
#    return np.sin(np.pi * x)




_MIN_L = 0.02

def kernel_derivatives(x, y, l, sigma):
    l = max(l, _MIN_L)
    r = x[:, None] - y[None, :]
    k = sigma**2 * np.exp(-0.5 * r**2 / l**2)
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


def rbf_kernel(x, y, l, sigma):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)



def gpr_build_K(x, l, sigma, sigma_n):
    K = rbf_kernel(x, x, l, sigma)
    K += sigma_n**2 * np.eye(len(x))
    return K


def gpr_nlml(x, y, l, sigma, sigma_n):
    try:
        K = gpr_build_K(x, l, sigma, sigma_n)
        K += 1e-8 * np.eye(len(x))
        sign, logdet = slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e10
        alpha  = solve(K, y)
        result = 0.5 * y @ alpha + 0.5 * logdet
        return result if np.isfinite(result) else 1e10
    except np.linalg.LinAlgError:
        return 1e10


def gpr_predict(x_train, y_train, x_star, l, sigma, sigma_n):
    K    = gpr_build_K(x_train, l, sigma, sigma_n) + 1e-8 * np.eye(len(x_train))
    Ks   = rbf_kernel(x_star, x_train, l, sigma)
    Kss  = rbf_kernel(x_star, x_star,  l, sigma)
    alpha = solve(K, y_train)
    mu    = Ks @ alpha
    v_mat = solve(K, Ks.T)
    var   = np.diag(Kss) - np.sum(Ks * v_mat.T, axis=1)
    return mu, np.maximum(var, 0.0)



def pigp_build_K(x_obs, c, l, sigma, sigma_n, v, kappa, with_bc=True):
    xf = c
    nb = len(x_obs)
    nc = len(c)
    nxc = nb + nc
    K   = np.zeros((nxc, nxc))

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


def pigp_build_K_star(x_star, x_obs, c, l, sigma, v, kappa):
    xf = c
    kbb, *_ = kernel_derivatives(x_star, x_obs, l, sigma)
    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x_star, xf, l, sigma)
    Kuf = v * k_xp - kappa * k_xpxp
    return np.hstack([kbb, Kuf])


def pigp_nlml(x_obs, c, y, l, sigma, sigma_n, v, kappa):
    try:
        K = pigp_build_K(x_obs, c, l, sigma, sigma_n, v, kappa)
        K += 1e-6 * np.eye(K.shape[0])
        sign, logdet = slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e10
        alpha  = solve(K, y)
        result = 0.5 * y @ alpha + 0.5 * logdet
        return result if np.isfinite(result) else 1e10
    except np.linalg.LinAlgError:
        return 1e10


def pigp_predict(x_obs, c, y, x_star, l, sigma, sigma_n, v, kappa):
    K    = pigp_build_K(x_obs, c, l, sigma, sigma_n, v, kappa) + 1e-6 * np.eye(len(x_obs) + len(c))
    Kuz  = pigp_build_K_star(x_star, x_obs, c, l, sigma, v, kappa)
    Kuu  = rbf_kernel(x_star, x_star, l, sigma)
    alpha = solve(K, y)
    mu    = Kuz @ alpha
    v_mat = solve(K, Kuz.T)
    var   = np.diag(Kuu) - np.sum(Kuz * v_mat.T, axis=1)
    return mu, np.maximum(var, 0.0)




def make_log_prior(prior_type, v_true=None, kappa_true=None):
    """
    Returns a function log_prior(eta) for the given prior type.
    eta = [log_l, log_sigma, log_sigma_n, log_v, log_kappa]
    """

    if prior_type == "physics":
        # Physics-derived: centred using energy / Pe-number arguments
        # Assumes weak prior knowledge that v, kappa ~ 0.1–1, Pe ~ O(1)
        mu    = np.array([np.log(0.3), np.log(0.5), np.log(0.01), np.log(0.3), np.log(0.3)])
        sigma = np.array([0.8,         0.5,          1.0,           0.6,         0.6        ])

    elif prior_type == "noninformative":
        # Weakly informative: centred at generic values, large variance
        mu    = np.array([0.0,   0.0,  np.log(0.01), 0.0,  0.0 ])
        sigma = np.array([1.5,   1.5,  1.5,           1.5,  1.5 ])

    elif prior_type == "misspecified":
        # Wrong: mass centred at v~5, kappa~5 – far from true (0.3, 0.6)
        mu    = np.array([np.log(0.3), np.log(0.5), np.log(0.01), np.log(5.0), np.log(5.0)])
        sigma = np.array([0.8,         0.5,          1.0,           0.3,         0.3        ])

    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    def log_prior(eta):
        return -0.5 * np.sum(((eta - mu) / sigma) ** 2)

    return log_prior, mu, sigma



def optimise_gpr(x_train, y_train, log_prior_fn, theta0=None):
    """
    Returns optimised (l, sigma, sigma_n) for pure GPR.
    theta0 in *log* space: [log_l, log_sigma, log_sigma_n]
    """
    if theta0 is None:
        theta0 = np.array([np.log(0.3), np.log(0.5), np.log(0.01)])

    def objective(eta):
        l, sigma, sigma_n = np.exp(eta[:3])
        nll = gpr_nlml(x_train, y_train, l, sigma, sigma_n)
        # only use the first 3 components of the prior (the GP kernel priors)
        lp  = -0.5 * np.sum(((eta[:3] - log_prior_fn.__closure__[0].cell_contents[:3])
                              / log_prior_fn.__closure__[1].cell_contents[:3]) ** 2)
        return nll - lp

    # simpler: reconstruct objective inline
    return objective, theta0


def map_optimise_gpr(x_train, y_train, log_prior_fn, prior_mu, prior_sigma, theta0=None):
    if theta0 is None:
        theta0 = prior_mu[:3].copy()

    def objective(eta):
        l, sigma, sigma_n = np.exp(eta)
        nll = gpr_nlml(x_train, y_train, l, sigma, sigma_n)
        lp  = -0.5 * np.sum(((eta - prior_mu[:3]) / prior_sigma[:3]) ** 2)
        return nll - lp

    t0  = time.time()
    res = minimize(objective, theta0, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-10})
    elapsed = time.time() - t0
    l, sigma, sigma_n = np.exp(res.x)
    return {"l": l, "sigma": sigma, "sigma_n": sigma_n,
            "nlml": res.fun, "time": elapsed, "success": res.success}


def map_optimise_pigp(x_obs, c, y, log_prior_fn, prior_mu, prior_sigma, theta0=None):
    if theta0 is None:
        theta0 = prior_mu.copy()

    def objective(eta):
        l, sigma, sigma_n, v, kappa = np.exp(eta)
        nll = pigp_nlml(x_obs, c, y, l, sigma, sigma_n, v, kappa)
        lp  = -0.5 * np.sum(((eta - prior_mu) / prior_sigma) ** 2)
        return nll - lp

    t0  = time.time()
    res = minimize(objective, theta0, method="L-BFGS-B",
                   bounds=[(-5, 3)] * 5,
                   options={"maxiter": 200, "ftol": 1e-10})
    elapsed = time.time() - t0
    l, sigma, sigma_n, v, kappa = np.exp(res.x)
    return {"l": l, "sigma": sigma, "sigma_n": sigma_n,
            "v": v, "kappa": kappa,
            "nlml": res.fun, "time": elapsed, "success": res.success}




def make_data(seed=42, n_int=40, nc=15, a=0.0, b=1.0,
              kappa_true=0.08, v_true=1.5, noise=1e-3):
    rng = np.random.default_rng(seed)
    x_int   = np.sort(rng.uniform(a, b, n_int))
    c       = np.sort(rng.uniform(a, b, nc))
    x_obs   = np.hstack([a, b, x_int])       
    yu      = u_exact_complex(x_int, kappa_true, v_true) + rng.normal(0, noise, n_int)
    yb      = np.array([0.0, 0.0])
    yf      = f(c) + rng.normal(0, 0.1, nc)
    y_pigp  = np.hstack([yb, yu, yf])
    y_gpr   = np.hstack([yb, yu])              # GPR only sees observations

    x_star  = np.sort(rng.uniform(a, b, 60))
    yt      = u_exact_complex(x_star, kappa_true, v_true)
    return x_obs, c, y_pigp, y_gpr, x_star, yt



def run_experiment(seed=42, n_int=40, nc=15,
                   kappa_true=0.08, v_true=1.5):
    prior_names = ["physics", "noninformative", "misspecified"]
    method_names = ["GPR", "PI-GP"]

    x_obs, c, y_pigp, y_gpr, x_star, yt = make_data(
        seed=seed, n_int=n_int, nc=nc,
        kappa_true=kappa_true, v_true=v_true
    )

    results = {}

    for prior_name in prior_names:
        log_prior_fn, prior_mu, prior_sigma = make_log_prior(
            prior_name, v_true=v_true, kappa_true=kappa_true
        )

        # --- GPR ---
        gpr_res = map_optimise_gpr(
            x_obs, y_gpr[:len(x_obs)],      # obs only (no collocation targets)
            log_prior_fn, prior_mu, prior_sigma
        )
        mu_gpr, var_gpr = gpr_predict(
            x_obs, y_gpr[:len(x_obs)], x_star,
            gpr_res["l"], gpr_res["sigma"], gpr_res["sigma_n"]
        )
        gpr_res["rmse"] = np.sqrt(np.mean((mu_gpr - yt)**2))
        gpr_res["mu"]   = mu_gpr
        gpr_res["var"]  = var_gpr
        results[(prior_name, "GPR")] = gpr_res

        # --- PI-GP ---
        pigp_res = map_optimise_pigp(
            x_obs, c, y_pigp,
            log_prior_fn, prior_mu, prior_sigma
        )
        mu_pi, var_pi = pigp_predict(
            x_obs, c, y_pigp, x_star,
            pigp_res["l"], pigp_res["sigma"], pigp_res["sigma_n"],
            pigp_res["v"],  pigp_res["kappa"]
        )
        pigp_res["rmse"] = np.sqrt(np.mean((mu_pi - yt)**2))
        pigp_res["mu"]   = mu_pi
        pigp_res["var"]  = var_pi
        results[(prior_name, "PI-GP")] = pigp_res

    return results, x_obs, c, x_star, yt




def plot_results(results, x_obs, c, x_star, yt,
                 kappa_true=0.08, v_true=1.5,
                 save_path="C:\\Users\\konst\\Desktop\\TU Delft\\Thesis\\Cld"):
    prior_names  = ["physics", "noninformative", "misspecified"]
    method_names = ["GPR", "PI-GP"]
    prior_labels = {"physics": "Physics-derived", "noninformative": "Non-informative",
                    "misspecified": "Misspecified"}
    colours      = {"GPR": "#1f77b4", "PI-GP": "#d62728"}

    ax_x = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=False)
    fig.suptitle(
        f"GPR vs PI-GP under different priors   (v_true={v_true}, κ_true={kappa_true})",
        fontsize=13
    )

    for row, prior_name in enumerate(prior_names):
        for col, method in enumerate(method_names):
            ax = axes[row, col]
            res = results[(prior_name, method)]
            mu, var = res["mu"], res["var"]
            std = np.sqrt(np.maximum(var, 0))

            ax.fill_between(x_star, mu - 2*std, mu + 2*std,
                            color=colours[method], alpha=0.18, label="±2σ")
            ax.plot(x_star, mu, color=colours[method], lw=2,
                    label=f"{method} (RMSE={res['rmse']:.4f})")
            ax.plot(ax_x, u_exact_complex(ax_x, kappa_true, v_true),
                    "k--", lw=1.2, label="Exact")
            ax.scatter(x_obs, yu[: len(x_obs)] if False else
                       [u_exact_complex(xi, kappa_true, v_true) for xi in x_obs],
                       c="orange", s=15, zorder=5, label="Observations")

            # print recovered v, kappa for PI-GP
            if method == "PI-GP":
                title_extra = (f" | v̂={res['v']:.3f} κ̂={res['kappa']:.3f}")
            else:
                title_extra = ""
            ax.set_title(
                f"{prior_labels[prior_name]} – {method}{title_extra}", fontsize=10
            )
            ax.set_ylim(-0.5, 0.55)
            if col == 0:
                ax.set_ylabel("u(x)")
            if row == 2:
                ax.set_xlabel("x")
            ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def print_summary(results, kappa_true=0.08, v_true=1.5):
    prior_names  = ["physics", "noninformative", "misspecified"]
    method_names = ["GPR", "PI-GP"]

    header = f"{'Prior':<18} {'Method':<8} {'RMSE':>8} {'Time(s)':>9} {'NLML':>10}"
    print("\n" + header)
    print("-" * len(header))

    for prior_name in prior_names:
        for method in method_names:
            r = results[(prior_name, method)]
            row = f"{prior_name:<18} {method:<8} {r['rmse']:8.5f} {r['time']:9.3f} {r['nlml']:10.2f}"
            if method == "PI-GP":
                row += f"   v̂={r['v']:.3f} (true {v_true})  κ̂={r['kappa']:.3f} (true {kappa_true})"
            print(row)
        print()




if __name__ == "__main__":
    print("Running GPR vs PI-GP prior comparison ...")
    print("  n_interior=40, n_collocation=15, domain=[0,1]")
    print("  true v=1.5, true κ=0.08\n")

    results, x_obs, c, x_star, yt = run_experiment(
        seed=0, n_int=40, nc=15, kappa_true=0.08, v_true=1.5
    )

    print_summary(results, kappa_true=0.08, v_true=1.5)
    plot_results(results, x_obs, c, x_star, yt,
                 kappa_true=0.08, v_true=1.5)

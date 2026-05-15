"""
Advection-diffusion with forcing functions on [0, 1]
-----------------------------------------------------
Steady-state 1D BVP:

    -κ u'' + μ u' = f(x)    on [0, 1]
    u(0) = 0,  u(1) = 0          (homogeneous Dirichlet)

Six forcing categories are tested against three representative (κ, μ) pairs
that span the diffusion-dominated, transitional, and advection-dominated
regimes (Pe = μ/κ = 0.01, 1, 100).

For each combination the script plots:
  • u(x)  — the solution (solid line)
  • f(x)  — the forcing function (shaded band, right-hand y-axis)

Forcing functions
-----------------
1. Uniform          f = 1                         constant interior source
2. Linear ramp      f = 2x − 1                    sign-changing linear source
3. Gaussian pulse   f = exp(−(x−0.3)²/0.02)      narrow interior source at x=0.3
4. Step (Heaviside) f = H(x − 0.5)               half-domain source switch
5. Sinusoidal       f = sin(2πx)                  single-mode oscillatory source
6. Multi-sine       f = sin(2πx)+0.5·sin(6πx)    multi-scale oscillatory source
7. Exponential wall f = exp(−x/0.05)              sharp near-inlet source
8. Double Gaussian  f = G(0.25) − G(0.75)         opposing source/sink pair
"""
#FIX RIGHT SIDE OF PLOTS!!!!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Forcing catalogue ─────────────────────────────────────────────────────────

def f_uniform(x):       return np.ones_like(x)
def f_linear(x):        return 2 * x - 1
def f_gaussian(x):      return np.exp(-((x - 0.3) ** 2) / 0.02)
def f_step(x):          return (x >= 0.5).astype(float)
def f_sine(x):          return np.sin(2 * np.pi * x)
def f_multisine(x):     return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
def f_expwall(x):       return np.exp(-x / 0.05)
def f_doublegauss(x):
    return (np.exp(-((x - 0.25) ** 2) / 0.005)
          - np.exp(-((x - 0.75) ** 2) / 0.005))

FORCINGS = [
    ("uniform",        f_uniform,     "f = 1"),
    ("linear ramp",    f_linear,      "f = 2x − 1"),
    ("Gaussian pulse", f_gaussian,    "f = exp(−(x−0.3)²/0.02)"),
    ("step (H)",       f_step,        "f = H(x − 0.5)"),
    ("sinusoidal",     f_sine,        "f = sin(2πx)"),
    ("multi-sine",     f_multisine,   "f = sin(2πx)+½sin(6πx)"),
    ("exp wall",       f_expwall,     "f = exp(−x/0.05)"),
    ("double Gaussian",f_doublegauss, "f = G(0.25) − G(0.75)"),
]

# ── Parameter regimes ─────────────────────────────────────────────────────────
# Three (κ, μ) pairs chosen to give Pe = 0.01, 1, 100.

PARAMS = [
    (1.0,   1.0, "diffusion-dom.\nPe = 1"),
    (0.1,   1,  "transition\nPe = 10"),
    (0.01,  100.0,  "advection-dom.\nPe = 10000"),
]

PARAM_COLORS = ["#3266AD", "#2E9E6B", "#C0392B"]

# ── Solver ────────────────────────────────────────────────────────────────────

def thomas(a, b, c, d):
    """In-place Thomas algorithm. Returns solution vector."""
    n  = len(d)
    b, c, d = b.copy(), c.copy(), d.copy()
    for i in range(1, n):
        m     = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    sol = np.zeros(n)
    sol[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        sol[i] = (d[i] - c[i] * sol[i + 1]) / b[i]
    return sol


def solve(kappa, mu, forcing_fn, N=600):
    """
    Solve  -kappa*u'' + mu*u' = f(x)  on [0,1], u(0)=u(1)=0.

    Diffusion : central differences  O(h²)
    Advection : upwind               O(h)   — unconditionally stable
    """
    h  = 1.0 / (N + 1)
    xi = np.linspace(h, 1 - h, N)          # interior nodes
    f  = forcing_fn(xi)

    d_diff =  2 * kappa / h**2
    o_diff = -1 * kappa / h**2

    if mu >= 0:                             # backward (upwind) difference
        d_adv =  mu / h
        l_adv = -mu / h
        u_adv =  0.0
    else:                                   # forward (upwind) difference
        d_adv = -mu / h
        l_adv =  0.0
        u_adv =  mu / h

    main  = np.full(N, d_diff + d_adv)
    lower = np.full(N - 1, o_diff + l_adv)
    upper = np.full(N - 1, o_diff + u_adv)

    rhs      = f.copy()
    rhs[0]  -= (o_diff + l_adv) * 0.0      # u(0) = 0
    rhs[-1] -= (o_diff + u_adv) * 0.0      # u(1) = 0

    u_inner = thomas(lower, main, upper, rhs)
    x_full  = np.concatenate([[0.0], xi, [1.0]])
    u_full  = np.concatenate([[0.0], u_inner, [0.0]])
    return x_full, u_full


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure():
    n_forcing = len(FORCINGS)
    n_params  = len(PARAMS)

    fig = plt.figure(figsize=(20, 3.6 * n_forcing + 5.8))
    #fig.suptitle(
    #    r"Advection-diffusion  $-\kappa\,u'' + \mu\,u' = f(x)$,"
    #    r"  $u(0)=u(1)=0$  — forcing × Pe regime",
    #    fontsize=12, y=1.005,
    #)

    gs = gridspec.GridSpec(n_forcing, n_params,
                           figure=fig, hspace=0.72, wspace=0.38)

    x_fine = np.linspace(0, 1, 800)

    for row, (fname, ffn, fexpr) in enumerate(FORCINGS):
        f_fine = ffn(x_fine)
        f_max  = np.max(np.abs(f_fine)) or 1.0   # for normalisation

        for col, (kappa, mu, plabel) in enumerate(PARAMS):
            ax = fig.add_subplot(gs[row, col])
            ax2 = ax.twinx()                      # right axis for f(x)

            # --- forcing band (right axis, behind) ---
            ax2.fill_between(x_fine, 0, f_fine,
                             alpha=0.13, color="#888888", linewidth=0)
            ax2.plot(x_fine, f_fine,
                     color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
            #ax2.set_ylim(-1.6 * f_max, 1.6 * f_max)
            ax2.axhline(0, color="#cccccc", linewidth=0.4)
            ax2.tick_params(labelsize=6.5, pad=2)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if col == n_params - 1:
                ax2.set_ylabel("f(x)", fontsize=7, color="#888888",
                               rotation=270, labelpad=10)
            else:
                ax2.set_yticklabels([])

            # --- solution (left axis) ---
            x, u = solve(kappa, mu, ffn)
            ax.plot(x, u, color=PARAM_COLORS[col], linewidth=1.8, zorder=3)
            ax.axhline(0, color="gray", linewidth=0.3, linestyle=":")

            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=6.5, pad=2)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Boundary-layer shade
            Pe = mu / kappa
            if Pe > 10:
                delta = kappa / mu
                ax.axvspan(1 - 5 * delta, 1, alpha=0.08,
                           color=PARAM_COLORS[col])

            # Column header (top row only)
            if row == 0:
                ax.set_title(plabel, fontsize=8, pad=4)

            # Row label (left column only)
            if col == 0:
                ax.set_ylabel(fname, fontsize=8, labelpad=4)

            # Forcing expression inside each subplot
            ax.text(0.5, 0.97, fexpr,
                    transform=ax.transAxes,
                    fontsize=6, va="top", ha="center",
                    color="#555555",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="none", alpha=0.75))

            ax.set_xlabel("x", fontsize=7, labelpad=2)

    # Column colour legend at top
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2, label=lbl.replace("\n", "  "))
        for c, (_, _, lbl) in zip(PARAM_COLORS, PARAMS)
    ]
    handles.append(
        plt.Line2D([0], [0], color="#888888", linewidth=1.2,
                   linestyle="--", alpha=0.7, label="f(x)  (right axis)")
    )
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.028),
               ncol=len(handles), fontsize=8, frameon=False)

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'Forcing':>20}  {'κ':>6}  {'μ':>6}  {'Pe':>8}  u_max")
    print("-" * 56)

    x_fine = np.linspace(0, 1, 800)
    for fname, ffn, _ in FORCINGS:
        for kappa, mu, _ in PARAMS:
            _, u = solve(kappa, mu, ffn)
            Pe   = mu / kappa
            print(f"{fname:>20}  {kappa:>6.3f}  {mu:>6.3f}  {Pe:>8.2g}  {np.max(np.abs(u)):.4f}")
        print()

    fig = make_figure()
    out = "advdiff_forcing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.show()

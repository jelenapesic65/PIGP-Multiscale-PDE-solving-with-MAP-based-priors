
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def solve_adv_diff(kappa: float, mu: float, N: int = 400) -> tuple[np.ndarray, np.ndarray]:

    h = 1.0 / (N + 1)
    x = np.linspace(0, 1, N + 2)           # full grid including BCs

    diag_diff =  2 * kappa / h**2
    off_diff  = -1 * kappa / h**2

    if mu >= 0:                             # upwind: use backward difference
        diag_adv =  mu / h
        off_lo   = -mu / h
        off_hi   =  0.0
    else:                                   # negative advection: forward difference
        diag_adv = -mu / h
        off_lo   =  0.0
        off_hi   =  mu / h                  # note: still negative mu

    main  = np.full(N, diag_diff + diag_adv)
    lower = np.full(N - 1, off_diff + off_lo)
    upper = np.full(N - 1, off_diff + off_hi)

    rhs = np.zeros(N)
    rhs[0]  -= (off_diff + off_lo) * 0.0   # u(0) = 0 → contributes 0
    rhs[-1] -= (off_diff + off_hi) * 1.0   # u(1) = 1

    u_inner = _thomas(lower, main, upper, rhs)

    u = np.concatenate([[0.0], u_inner, [1.0]])
    return x, u


def _thomas(a: np.ndarray, b: np.ndarray,
            c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(d)
    b, d = b.copy(), d.copy()
    c = c.copy()
    for i in range(1, n):
        m    = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x




KAPPA_VALUES = [1.0, 1e-2, 1e-4]        
MU_VALUES    = [0.0, 0.01, 1.0, 100.0]   

REGIME_LABELS = {
    (1.0,    0.0 ): "pure diffusion\n(Pe = 0)",
    (1.0,    0.01): "diffusion-dom.\n(Pe = 0.01)",
    (1.0,    1.0 ): "Pe ~ 1\n(transition)",
    (1.0,   100.0): "mild advection\n(Pe = 100)",
    (1e-2,   0.0 ): "pure diffusion\n(Pe = 0)",
    (1e-2,   0.01): "diffusion-dom.\n(Pe = 1)",
    (1e-2,   1.0 ): "Pe ~ 100\n(transition)",
    (1e-2,  100.0): "advection-dom.\n(Pe = 1e4)",
    (1e-4,   0.0 ): "pure diffusion\n(Pe = 0)",
    (1e-4,   0.01): "Pe ~ 1e2",
    (1e-4,   1.0 ): "advection-dom.\n(Pe = 1e4)",
    (1e-4,  100.0): "near-hyperbolic\n(Pe = 1e6)",
}


def make_figure():
    nrows = len(KAPPA_VALUES)
    ncols = len(MU_VALUES)

    fig = plt.figure(figsize=(14, 9))
    #fig.suptitle(
    #    "1D advection-diffusion:  $-\\kappa\\,u'' + \\mu\\,u' = 0$"
    #    "  on $[0,1]$,  $u(0)=0$,  $u(1)=1$",
    #    fontsize=13, y=1.01,
    #)

    gs = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.55, wspace=0.35,
    )

    colours = ["#3266AD", "#2E9E6B", "#D47B22", "#C0392B"]

    for row, kappa in enumerate(KAPPA_VALUES):
        for col, mu in enumerate(MU_VALUES):
            ax = fig.add_subplot(gs[row, col])

            x, u = solve_adv_diff(kappa, mu)
            Pe = mu / kappa if kappa > 0 else float("inf")

            ax.plot(x, u, color=colours[col], linewidth=1.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            ax.axhline(1, color="gray", linewidth=0.4, linestyle="--")

            # Column headers (top row only)
            if row == 0:
                ax.set_title(
                    f"$\\mu = {mu}$",
                    fontsize=10, pad=4,
                )

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(
                    f"$\\kappa = {kappa:.0e}$",
                    fontsize=9, labelpad=4,
                )

            # Pe annotation
            pe_str = f"Pe = {Pe:.1g}" if not np.isinf(Pe) else "Pe = 0"
            regime = REGIME_LABELS.get((kappa, mu), "")
            ax.set_title(
                f"$\\mu={mu}$\n{pe_str}" if row == 0
                else f"{pe_str}",
                fontsize=8, pad=3,
            )

            # Regime text box
            ax.text(
                0.03, 0.97, regime,
                transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
            )

            ax.tick_params(labelsize=7)
            ax.set_xlabel("$x$", fontsize=7, labelpad=2)

            # Shade boundary-layer region for high-Pe cases
            if Pe > 10:
                # Estimate BL thickness δ ~ kappa/mu
                delta = kappa / mu if mu > 0 else 0
                ax.axvspan(1 - 5 * delta, 1, alpha=0.12, color=colours[col])

    # Column super-headers
    for col, mu in enumerate(MU_VALUES):
        fig.text(
            0.13 + col * 0.215, 1.005,
            f"$\\mu = {mu}$",
            ha="center", fontsize=10, fontweight="bold",
        )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Solving advection-diffusion for all (kappa, mu) combinations…\n")

    # Print regime summary table
    header = f"{'kappa':>10}  {'mu':>8}  {'Pe':>12}  {'regime'}"
    print(header)
    print("-" * len(header))
    for kappa in KAPPA_VALUES:
        for mu in MU_VALUES:
            Pe = mu / kappa if kappa > 0 and mu > 0 else (0 if mu == 0 else float("inf"))
            if Pe == 0:
                regime = "pure diffusion"
            elif Pe < 1:
                regime = "diffusion-dominated"
            elif Pe < 10:
                regime = "transition"
            elif Pe < 1000:
                regime = "advection-dominated"
            else:
                regime = "near-hyperbolic (BL)"
            print(f"{kappa:>10.1e}  {mu:>8.3g}  {Pe:>12.2g}  {regime}")
        print()

    fig = make_figure()
    out_path = "adv_diff_regimes.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.show()

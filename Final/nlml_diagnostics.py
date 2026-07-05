
#NLML diagnostics for the plain-GPR vs PIGP comparison

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.model_selection import KFold

import advdiff_2d_gpr_pigp as M

# main module
Xo, uo = M.xy_obs, M.u_obs_n            # sensors (standardised targets)
Xb, ub = M.xy_bc,  M.u_bc_n             # boundary points (u=0)
Xc, fc = M.xy_col, M.f_col_n           # collocation points, forcing
xs, ut = M.xy_star, M.u_true           # test grid, ground truth (physical units)
S, sn  = M.S, M.SN_FIX                  # target scale, fixed sensor noise
v, k   = M.V_TRUE, M.KAPPA_TRUE

#NLML / error
def pure_nlml(lx, ly, sg, phys):

    _, y, K, _ = M.build_system(Xo, uo, Xc, fc, Xb, ub, lx, ly, sg, sn, v, k, phys)
    try:
        c = cho_factor(K)
    except np.linalg.LinAlgError:
        return np.nan
    ld = 2.0 * np.log(np.diag(c[0])).sum()
    return 0.5 * (y @ cho_solve(c, y)) + 0.5 * ld

def profile_nlml(lx, ly, phys, sgs):
    #NLML profiled over the signal variance
    return np.nanmin([pure_nlml(lx, ly, s, phys) for s in sgs]) #find the minimum over the signal variance grid? 

def relL2(lx, ly, sg, phys):
    mu, _ = M.gp_posterior(Xo, uo, Xc, fc, Xb, ub, xs, lx, ly, sg, sn, v, k, phys)
    return np.linalg.norm(mu*S - ut) / np.linalg.norm(ut)

# per-model figure
def diagnosis_figure(phys, opt, fname, label, color):
    """3-panel NLML diagnosis for one model (phys=False GPR, True PIGP)."""
    lxo, lyo, sgo = opt['lx'], opt['ly'], opt['sigma']
    sgs = np.logspace(np.log10(0.2), np.log10(3.0), 10)
    L = np.logspace(np.log10(0.015), np.log10(0.5), 40)

    # 2D profiled landscape (rows=ly, cols=lx)
    Z = np.array([[profile_nlml(lx, ly, phys, sgs) for lx in L] for ly in L])
    # 1D slices through the optimum
    line = np.logspace(np.log10(0.012), np.log10(0.5), 60)
    n_lx = np.array([profile_nlml(lx, lyo, phys, sgs) for lx in line])
    r_lx = np.array([relL2(lx, lyo, sgo, phys) for lx in line])
    n_ly = np.array([profile_nlml(lxo, ly, phys, sgs) for ly in line])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0), constrained_layout=True)
    Zp = np.log10(Z - np.nanmin(Z) + 1.0)
    im = ax[0].pcolormesh(L, L, Zp, cmap='viridis', shading='auto')
    ax[0].contour(L, L, Zp, levels=8, colors='w', linewidths=0.5, alpha=0.5)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].scatter([lxo], [lyo], c='red', s=90, edgecolors='k', zorder=5, label='optimum')
    ax[0].set_xlabel('lx'); ax[0].set_ylabel('ly'); ax[0].legend(fontsize=8, loc='lower right')
    ax[0].set_title(f'(a) {label} NLML landscape\nlog10(NLML - min + 1), profiled over sigma',
                    fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    a, a2 = ax[1], ax[1].twinx()
    a.semilogy(line, n_lx - np.nanmin(n_lx) + 1, color=color, lw=2, label='NLML (shifted)')
    a2.plot(line, r_lx, color='#e94560', lw=2, label='relL2 error')
    a.axvline(line[np.nanargmin(n_lx)], ls='--', color=color, alpha=.7)
    a2.axvline(line[np.argmin(r_lx)], ls='--', color='#e94560', alpha=.7)
    a.set_xscale('log'); a.set_xlabel('lx  (ly = ly*)')
    a.set_ylabel('NLML - min + 1  (log)', color=color)
    a2.set_ylabel('relative L2 error', color='#e94560')
    nlml_min_lx = line[np.nanargmin(n_lx)]
    tag = 'min at boundary (lx->0)' if nlml_min_lx <= line[2] else f'interior min lx={nlml_min_lx:.3f}'
    a.set_title(f'(b) NLML vs lx: {tag}\nrelL2-optimal lx={line[np.argmin(r_lx)]:.3f}',
                fontsize=10, fontweight='bold')
    ln = a.get_lines()[:1] + a2.get_lines()[:1]
    a.legend(ln, [l.get_label() for l in ln], fontsize=8, loc='upper center')

    ax[2].semilogy(line, n_ly - np.nanmin(n_ly) + 1, color=color, lw=2)
    ax[2].axvline(line[np.nanargmin(n_ly)], ls='--', color=color, alpha=.7)
    ax[2].set_xscale('log'); ax[2].set_xlabel('ly  (lx = lx*)')
    ax[2].set_ylabel('NLML - min + 1  (log)', color=color)
    ax[2].set_title(f'(c) NLML vs ly: interior min ly={line[np.nanargmin(n_ly)]:.3f}',
                    fontsize=10, fontweight='bold')

    fig.suptitle(f'{label} marginal likelihood diagnosis  (CASE={M.CASE})',
                 fontsize=12, fontweight='bold')
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    print(f'saved {fname}')

# fixes figure
def cv_select(grid, nfold=5, seed=0):
    """5-fold CV over the sensors: for each (lx,ly) on the grid, average the
    held-out MSE over 5 train/predict splits; return the best (lx,ly)."""
    kf = KFold(nfold, shuffle=True, random_state=seed)
    def pred(Xtr, ytr, Xte, lx, ly):
        K = M.k2d(Xtr, Xtr, lx, ly, 1.0) + (sn**2 + 1e-8)*np.eye(len(Xtr))
        return M.k2d(Xte, Xtr, lx, ly, 1.0) @ np.linalg.solve(K, ytr)
    best = (np.inf, None)
    for lx in grid:
        for ly in grid:
            e = []
            for tr, te in kf.split(Xo):
                Xtr = np.vstack([Xb, Xo[tr]]); ytr = np.concatenate([ub, uo[tr]])
                e.append(np.mean((pred(Xtr, ytr, Xo[te], lx, ly) - uo[te])**2))
            if np.mean(e) < best[0]:
                best = (np.mean(e), (lx, ly))
    return best[1]

def map_fit(prior_mu, prior_w):
    """MAP fit of plain GPR with a strong log-normal prior on lx, ly."""
    def obj(t):
        lx, ly, sg = t
        val = pure_nlml(lx, ly, sg, False)
        if not np.isfinite(val): return 1e10
        val -= -0.5*((np.log(lx)-np.log(prior_mu))/prior_w)**2
        val -= -0.5*((np.log(ly)-np.log(prior_mu))/prior_w)**2
        val -= -0.5*((np.log(sg)-np.log(1.0))/0.7)**2
        return val
    r = minimize(obj, [prior_mu, prior_mu, 1.0], method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=1e-3, maxiter=500))
    return r.x

def fixes_figure(fname):
    sgs = np.logspace(np.log10(0.2), np.log10(3.0), 10)
    line = np.logspace(np.log10(0.012), np.log10(0.5), 60)
    lyo = M.PIGP['ly']
    n_lx = np.array([profile_nlml(lx, lyo, False, sgs) for lx in line])
    r_lx = np.array([relL2(lx, lyo, M.PIGP['sigma'], False) for lx in line])

    ml2 = (M.PIGP['lx'], relL2(M.PIGP['lx'], M.PIGP['ly'], M.PIGP['sigma'], False))
    mlx, mly, msg = map_fit(0.2, 0.15)
    mp = (mlx, relL2(mlx, mly, msg, False))
    cvx, cvy = cv_select(np.logspace(np.log10(0.03), np.log10(0.4), 12))
    cv = (cvx, relL2(cvx, cvy, 1.0, False))
    pigp_rel = relL2(M.PIGP['lx'], M.PIGP['ly'], M.PIGP['sigma'], True)

    fig, a = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    a2 = a.twinx()
    a.semilogy(line, n_lx - np.nanmin(n_lx) + 1, color='#2a4d9b', lw=2, label='PIGP NLML (shifted)')
    a2.plot(line, r_lx, color='#e94560', lw=2, label='PIGP relL2 vs lx')
    a2.scatter(*ml2, c='k', s=80, zorder=6, label=f'ML-II  (relL2={ml2[1]:.2f})')
    a2.scatter(*mp, c='orange', s=80, zorder=6, label=f'strong prior  (relL2={mp[1]:.2f})')
    a2.scatter(*cv, c='green', s=80, zorder=6, label=f'5-fold CV  (relL2={cv[1]:.2f})')
    a2.axhline(pigp_rel, ls=':', color='purple', lw=2, label=f'PIGP  (relL2={pigp_rel:.2f})')
    a.set_xscale('log'); a.set_xlabel('lx')
    a.set_ylabel('NLML - min + 1  (log)', color='#2a4d9b')
    a2.set_ylabel('relative L2 error', color='#e94560')
    a.set_title('Taming the PIGP NLML pathology: ML-II vs strong prior vs CV\n'
                '(NLML min sits at lx->0; CV recovers the accurate lengthscale)',
                fontsize=11, fontweight='bold')
    ln = a.get_lines()[:1] + a2.get_lines()[:1] + a2.collections
    a2.legend(loc='center right', fontsize=8)
    a.legend(loc='lower left', fontsize=8)
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    print(f'saved {fname}')
    print(f'  ML-II lx={ml2[0]:.3f} relL2={ml2[1]:.3f} | strong-prior lx={mp[0]:.3f} relL2={mp[1]:.3f} '
          f'| CV lx={cvx:.3f},ly={cvy:.3f} relL2={cv[1]:.3f} | PIGP relL2={pigp_rel:.3f}')

# ============================================================ run
if __name__ == '__main__':
    diagnosis_figure(False, M.GPR,  'fig_gpr_nlml_diagnosis.png',  'plain GPR', '#2a4d9b')
    diagnosis_figure(True,  M.PIGP, 'fig_pigp_nlml_diagnosis.png', 'PIGP',      '#1b7f4d')
    fixes_figure('fig_nlml_fixes.png')
    print('done.')

"""
Robust MLE search for the two-scale GP problem.
Multiple optimizers, multiple seeds, full l sweep plotted.
"""
import numpy as np
from numpy.linalg import solve, slogdet
from scipy.optimize import minimize, differential_evolution
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

np.random.seed(0)

Pe = 500
sigma = 1.0
sn = 1e-3

def u_true(x):
    #return np.sin(2 * np.pi * x) * (1 - np.exp(-Pe * (1 - x)))
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1) + np.sin(np.pi*x)
n_s = 25
n_d = 10
n   = n_s + n_d

x_s   = np.linspace(0.00, 0.90, n_s)
x_d   = np.linspace(0.92, 1.00, n_d)
x_all = np.concatenate([x_s, x_d])
y_all = np.concatenate([u_true(x_s), u_true(x_d)])

l_s = (x_s[-1] - x_s[0]) / (n_s - 1)
l_d = (x_d[-1] - x_d[0]) / (n_d - 1)

def rbf(x, y, l):
    r2 = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * r2 / l**2)

def nlml(l_raw):
    """NLML as function of log(l) for unconstrained optimisation."""
    l = np.exp(l_raw)
    K = rbf(x_all, x_all, l) + sn**2 * np.eye(n)
    try:
        alpha = solve(K, y_all)
        sign, ld = slogdet(K)
        if sign <= 0:
            return 1e10
        return 0.5 * (y_all @ alpha + ld)
    except Exception:
        return 1e10

def nlml_l(l):
    """NLML as function of l directly (for grid sweep)."""
    if l <= 0:
        return 1e10
    return nlml(np.log(l))

# ── 1. Dense grid sweep ────────────────────────────────────────────────────────
l_grid = np.logspace(np.log10(l_d * 0.3), np.log10(0.5), 500)
nlml_grid = np.array([nlml_l(l) for l in l_grid])

l_best_grid = l_grid[np.argmin(nlml_grid)]

print("=" * 65)
print("1. DENSE GRID SWEEP (500 points, log-spaced)")
print("=" * 65)
print(f"   l_d (focused spacing) = {l_d:.5f}")
print(f"   l_s (static spacing)  = {l_s:.5f}")
print(f"   Grid minimum at l*    = {l_best_grid:.5f}   NLML = {min(nlml_grid):.4f}")
print()

# Print NLML at key reference points
for lname, lval in [("l_d", l_d), ("l_s", l_s), ("l* grid", l_best_grid)]:
    print(f"   NLML at {lname:8s} ({lval:.5f}) = {nlml_l(lval):.4f}")

# ── 2. Multiple random starts, L-BFGS-B ───────────────────────────────────────
print()
print("=" * 65)
print("2. L-BFGS-B  —  30 random starting points")
print("=" * 65)

seeds_lbfgs = []
np.random.seed(42)
for i in range(30):
    l0 = np.random.uniform(np.log(l_d * 0.5), np.log(0.4))
    res = minimize(nlml, l0, method='L-BFGS-B',
                   bounds=[(-10, 0)],
                   options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
    l_opt = np.exp(res.x[0])
    seeds_lbfgs.append((l_opt, res.fun, res.success))

l_vals = np.array([x[0] for x in seeds_lbfgs])
f_vals = np.array([x[1] for x in seeds_lbfgs])

print(f"   {'Run':>4} {'l0_init':>10} {'l*':>10} {'NLML':>10} {'success'}")
print(f"   {'-'*52}")
np.random.seed(42)
for i, (lopt, fopt, suc) in enumerate(seeds_lbfgs):
    l0 = np.exp(np.random.uniform(np.log(l_d*0.5), np.log(0.4)))
    print(f"   {i+1:>4} {l0:>10.5f} {lopt:>10.5f} {fopt:>10.4f} {'yes' if suc else 'no'}")

print()
print(f"   Best l*    = {l_vals[np.argmin(f_vals)]:.5f}   (NLML = {min(f_vals):.4f})")
print(f"   Median l*  = {np.median(l_vals):.5f}")
print(f"   Min l*     = {l_vals.min():.5f}")
print(f"   Max l*     = {l_vals.max():.5f}")
print(f"   Unique modes (within 1e-3):")
rounded = np.round(l_vals, 3)
unique, counts = np.unique(rounded, return_counts=True)
for u, c in zip(unique, counts):
    marker = " <-- global (grid)" if abs(u - l_best_grid) < 0.005 else ""
    print(f"     l* ~ {u:.5f}  ({c} runs){marker}")

# ── 3. Nelder-Mead (derivative-free) ─────────────────────────────────────────
print()
print("=" * 65)
print("3. Nelder-Mead  —  30 random starting points")
print("=" * 65)

seeds_nm = []
np.random.seed(99)
for i in range(30):
    l0 = np.random.uniform(np.log(l_d * 0.5), np.log(0.4))
    res = minimize(nlml, l0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    l_opt = np.exp(res.x[0])
    seeds_nm.append((l_opt, res.fun))

l_nm = np.array([x[0] for x in seeds_nm])
f_nm = np.array([x[1] for x in seeds_nm])
print(f"   Best l*    = {l_nm[np.argmin(f_nm)]:.5f}   (NLML = {min(f_nm):.4f})")
print(f"   Median l*  = {np.median(l_nm):.5f}")
print(f"   Unique modes (within 1e-3):")
rounded_nm = np.round(l_nm, 3)
unique_nm, counts_nm = np.unique(rounded_nm, return_counts=True)
for u, c in zip(unique_nm, counts_nm):
    print(f"     l* ~ {u:.5f}  ({c} runs)")

# ── 4. Differential evolution (global, derivative-free) ──────────────────────
print()
print("=" * 65)
print("4. Differential Evolution  (global optimizer)")
print("=" * 65)

res_de = differential_evolution(
    nlml, bounds=[(-10, 0)],
    seed=7, maxiter=2000, tol=1e-12, popsize=20, mutation=(0.5, 1.5)
)
l_de = np.exp(res_de.x[0])
print(f"   l* = {l_de:.5f}   NLML = {res_de.fun:.4f}   success = {res_de.success}")

# ── 5. NLML profile at key points ─────────────────────────────────────────────
print()
print("=" * 65)
print("5. NLML PROFILE at selected l values")
print("=" * 65)
print(f"   {'l':>10}  {'NLML':>10}  note")
print(f"   {'-'*45}")
check_ls = sorted(set(list(np.logspace(np.log10(l_d), np.log10(0.3), 20))
                      + [l_d, l_s, l_best_grid, l_de]))
for l in check_ls:
    v = nlml_l(l)
    note = ""
    if abs(l - l_d)   < 0.0002: note = "<- focused spacing"
    if abs(l - l_s)   < 0.002:  note = "<- static spacing"
    if abs(l - l_best_grid) < 0.003: note = "<- grid minimum"
    if abs(l - l_de)  < 0.003:  note = "<- DE minimum"
    print(f"   {l:10.5f}  {v:10.4f}  {note}")

# ── 6. Gradient profile ────────────────────────────────────────────────────────
print()
print("=" * 65)
print("6. NUMERICAL GRADIENT d(NLML)/dl  at selected l values")
print("   (finite difference, confirms sign-change location)")
print("=" * 65)
print(f"   {'l':>10}  {'NLML':>10}  {'d NLML/dl':>12}  note")
print(f"   {'-'*55}")
eps = 1e-6
prev_g = None
for l in np.logspace(np.log10(l_d * 0.8), np.log10(0.35), 35):
    v   = nlml_l(l)
    g   = (nlml_l(l + eps) - nlml_l(l - eps)) / (2 * eps)
    note = ""
    if prev_g is not None and prev_g * g < 0:
        note = "<-- minimum (sign change)"
    if abs(l - l_d) < 0.0004: note = "<- focused spacing"
    if abs(l - l_s) < 0.002:  note = "<- static spacing"
    print(f"   {l:10.5f}  {v:10.4f}  {g:12.4f}  {note}")
    prev_g = g

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  l_d (focused spacing) = {l_d:.5f}")
print(f"  l_s (static spacing)  = {l_s:.5f}")
print(f"  l* from dense grid    = {l_best_grid:.5f}")
print(f"  l* from L-BFGS best   = {l_vals[np.argmin(f_vals)]:.5f}")
print(f"  l* from Nelder-Mead   = {l_nm[np.argmin(f_nm)]:.5f}")
print(f"  l* from Diff. Evol.   = {l_de:.5f}")
print()
print(f"  NLML at l_d  = {nlml_l(l_d):.4f}")
print(f"  NLML at l_s  = {nlml_l(l_s):.4f}")
print(f"  NLML at l*   = {min(nlml_grid):.4f}")
print()
print("  Interpretation:")
rel_d = (nlml_l(l_d) - min(nlml_grid))
rel_s = (nlml_l(l_s) - min(nlml_grid))
print(f"  NLML(l_d) - NLML(l*) = {rel_d:.4f}  "
      f"({'worse' if rel_d>0 else 'better'} than optimum)")
print(f"  NLML(l_s) - NLML(l*) = {rel_s:.4f}  "
      f"({'worse' if rel_s>0 else 'better'} than optimum)")


import matplotlib.pyplot as plt

# Use best lengthscale (grid is safest/global)
l_opt = l_best_grid

# Prediction grid (high resolution)
x_star = np.linspace(0, 1, 1000)

# Kernel blocks
K   = rbf(x_all, x_all, l_opt) + sn**2 * np.eye(n)
Ks  = rbf(x_all, x_star, l_opt)
Kss = rbf(x_star, x_star, l_opt)

# Solve for posterior
alpha = solve(K, y_all)
mu    = Ks.T @ alpha

v     = solve(K, Ks)
cov   = Kss - Ks.T @ v
std   = np.sqrt(np.clip(np.diag(cov), 0, None))

# True function
y_true = u_true(x_star)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x_star, y_true, 'k--', label='True function')
plt.plot(x_star, mu, 'b', label='GP posterior mean')
plt.fill_between(x_star, mu - 2*std, mu + 2*std,
                 alpha=0.3, label='±2 std')

plt.scatter(x_s, u_true(x_s), c='red', s=25, label='Static points')
plt.scatter(x_d, u_true(x_d), c='green', s=25, label='Boundary points')

plt.title(f"GP posterior (l* = {l_opt:.5f})")
plt.legend()
plt.grid(True)
plt.show()

# Higher resolution NLML sweep
l_grid_hr = np.logspace(np.log10(l_d * 0.3), np.log10(0.5), 2000)
nlml_hr   = np.array([nlml_l(l) for l in l_grid_hr])

plt.figure(figsize=(10, 5))
plt.plot(l_grid_hr, nlml_hr, 'b')

# Mark key points
plt.axvline(l_d, color='green', linestyle='--', label='l_d (boundary spacing)')
plt.axvline(l_s, color='red', linestyle='--', label='l_s (static spacing)')
plt.axvline(l_best_grid, color='black', linestyle='-', label='l* (optimum)')
plt.axvspan(l_d*0.8, l_d*1.2, alpha=0.1, color='green')
plt.axvspan(l_s*0.8, l_s*1.2, alpha=0.1, color='red')
plt.xscale('log')
plt.xlabel('Lengthscale l')
plt.ylabel('NLML')
plt.title('NLML vs lengthscale (high resolution)')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()

def nlml_terms(l):
    """Return (data_fit, log_det, total NLML)"""
    if l <= 0:
        return np.nan, np.nan, 1e10
    
    K = rbf(x_all, x_all, l) + sn**2 * np.eye(n)
    
    try:
        alpha = solve(K, y_all)
        sign, ld = slogdet(K)
        if sign <= 0:
            return np.nan, np.nan, 1e10
        
        data_fit = 0.5 * (y_all @ alpha)
        log_det  = 0.5 * ld
        total    = data_fit + log_det
        
        return data_fit, log_det, total
    
    except Exception:
        return np.nan, np.nan, 1e10
    
    # High-resolution decomposition

def grad_fd(l, eps=1e-6):
    return (nlml_l(l + eps) - nlml_l(l - eps)) / (2 * eps)

def grad_terms_fd(l, eps=1e-6):
    d1, log1, _ = nlml_terms(l + eps)
    d2, log2, _ = nlml_terms(l - eps)
    
    g_data = (d1 - d2) / (2 * eps)
    g_log  = (log1 - log2) / (2 * eps)
    
    return g_data, g_log
    
l_grid_hr = np.logspace(np.log10(l_d * 0.3), np.log10(1), 1500)

data_terms = []
logdet_terms = []
total_terms = []
g_data_arr = []
g_log_arr  = []

for l in l_grid_hr:
    d, logd, tot = nlml_terms(l)
    data_terms.append(d)
    logdet_terms.append(logd)
    total_terms.append(tot)
    gd, gl = grad_terms_fd(l)
    g_data_arr.append(gd)
    g_log_arr.append(gl)


data_terms   = np.array(data_terms)
logdet_terms = np.array(logdet_terms)
total_terms  = np.array(total_terms)
g_data_arr = np.array(g_data_arr)
g_log_arr  = np.array(g_log_arr)

plt.figure(figsize=(10, 5))

plt.plot(l_grid_hr, total_terms, 'k', label='Total NLML', linewidth=2)
plt.plot(l_grid_hr, data_terms, 'b--', label='Data-fit term')
plt.plot(l_grid_hr, logdet_terms, 'r--', label='Log-det term')

# Key markers
plt.axvline(l_d, color='green', linestyle='--', label='l_d')
plt.axvline(l_s, color='orange', linestyle='--', label='l_s')
plt.axvline(l_best_grid, color='black', linestyle='-', label='l*')

plt.xscale('log')
plt.xlabel('Lengthscale l')
plt.ylabel('Value')
plt.title('NLML decomposition')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()

plt.figure()
plt.plot(l_grid_hr, logdet_terms)
plt.xscale('log')
plt.title("Log-det term only")
plt.grid()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(l_grid_hr, g_data_arr, label='grad data-fit', linewidth=2)
plt.plot(l_grid_hr, g_log_arr,  label='grad log-det', linewidth=2)

# Total gradient (optional but useful)
plt.plot(l_grid_hr, g_data_arr + g_log_arr, 'k--', label='total gradient')

# Reference lines
plt.axhline(0, color='black', linewidth=1)
plt.axvline(l_d, color='green', linestyle='--', label='l_d')
plt.axvline(l_s, color='orange', linestyle='--', label='l_s')
plt.axvline(l_best_grid, color='red', linestyle='--', label='l*')

plt.xscale('log')
plt.xlabel('Lengthscale l')
plt.ylabel('Gradient')
plt.title('Gradient decomposition of NLML')

plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(l_grid_hr, np.abs(g_data_arr), label='|grad data-fit|')
plt.plot(l_grid_hr, np.abs(g_log_arr),  label='|grad log-det|')

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Lengthscale l')
plt.ylabel('Gradient magnitude (log scale)')
plt.title('Gradient magnitude comparison')

plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()

for l in [l_d, l_s, 0.2, 0.5, 1.0]:
    gd, gl = grad_terms_fd(l)
    print(f"l={l:.4f} | grad_data={gd:.4f} | grad_logdet={gl:.4f}")


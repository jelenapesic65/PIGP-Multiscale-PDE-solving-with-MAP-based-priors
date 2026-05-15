import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve


def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :])**2
    return sigma**2 * np.exp(-0.5*sqdist/l**2)

def interp_weights_d1(x, U, h):
    W = np.zeros(len(U))
    j = np.searchsorted(U, x) - 1
    j = np.clip(j, 0, len(U)-2)
    W[j] = -1/h
    W[j+1] = 1/h
    return W

def interp_weights_d2(x, U, h):
    W = np.zeros(len(U))
    j = np.searchsorted(U, x) - 1
    j = np.clip(j, 1, len(U)-2)  # avoid boundary wrap
    W[j-1] = 1/h**2
    W[j]   = -2/h**2
    W[j+1] = 1/h**2
    return W

def build_K(u, x_internal, l, sigma, v, kappa, with_bc=False):
    m = len(u)
    n_internal = len(x_internal)
    Kuu = rbf_kernel(u, u, l, sigma)
    
    # PDE rows for internal points
    Wd1 = np.vstack([interp_weights_d1(xi, u, u[1]-u[0]) for xi in x_internal])
    Wd2 = np.vstack([interp_weights_d2(xi, u, u[1]-u[0]) for xi in x_internal])
    W_pde = v*Wd1 - kappa*Wd2
    
    if with_bc:
        # Boundary condition rows
        W_bc = np.zeros((2, m))
        W_bc[0, 0] = 1.0   # u(0)=0
        W_bc[1, -1] = 1.0  # u(1)=0
        W = np.vstack([W_bc, W_pde])
    else:
        W = W_pde
    
    return Kuu, W


a, b = 0.0, 1.0
m = 50        # inducing points
n_internal = 50

u = np.linspace(a, b, m)
x_internal = np.linspace(a, b, n_internal)[1:-1]  # exclude boundaries
h = u[1]-u[0]

v = 0.5
kappa = 0.5
l = 0.3
sigma = 1.0
sigma_n = 1e-3

# PDE forcing
f = lambda x: np.sin(np.pi*x)
y_internal = f(x_internal) + np.random.normal(0, 0.01, len(x_internal))


Kuu = rbf_kernel(u, u, l, sigma)
prior_mean = np.zeros_like(u)
prior_std = np.sqrt(np.diag(Kuu))


Kuu, W_internal = build_K(u, x_internal, l, sigma, v, kappa, with_bc=False)
K_internal = W_internal @ Kuu @ W_internal.T + sigma_n**2*np.eye(len(x_internal))
alpha_internal = solve(K_internal, y_internal)
posterior_internal = Kuu @ W_internal.T @ alpha_internal


y_full = np.hstack([0.0, 0.0, y_internal])
x_full = np.hstack([0.0, 1.0, x_internal])
Kuu, W_full = build_K(u, x_internal, l, sigma, v, kappa, with_bc=True)
K_full = W_full @ Kuu @ W_full.T + sigma_n**2*np.eye(len(y_full))
alpha_full = solve(K_full, y_full)
posterior_full = Kuu @ W_full.T @ alpha_full


plt.figure(figsize=(10,6))
plt.fill_between(u, prior_mean-2*prior_std, prior_mean+2*prior_std, color='gray', alpha=0.3, label='Prior 2σ')
plt.plot(u, posterior_internal, 'b', label='Posterior (internal points only)')
plt.plot(u, posterior_full, 'r', label='Posterior (internal + BCs)')
plt.plot(u, np.sin(np.pi*u)/ (np.pi**2), '--', label='Exact solution (for reference)')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Prior vs Posterior with/without Boundary Conditions")
plt.show()

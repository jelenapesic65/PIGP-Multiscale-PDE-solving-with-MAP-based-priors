import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import minimize
from numpy.linalg import slogdet, solve

def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def log_prior(theta):
    l, sigma, log_sigma_n,v,kappa = theta

    log_l = np.log(l)
    log_sigma = np.log(sigma) #force positivity of hyperparameters

    # Hyperprior parameters
    #the bounds are made wrt to the fact that the domain is [0,1] and the expected behavior of the equation

    mu_l, std_l = np.log(0.3), 0.5 #multiplicative deviations
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

a,b = 0.0, 1.0
n1, n2 = 100, 100
m1, m2 = 30, 30
N = m1*m2
x = np.random.uniform(a, b, n1)
y = np.random.uniform(a, b, n2)

Xg, Yg = np.meshgrid(x, y, indexing="ij")
yf = f(Xg, Yg).ravel()

ux = np.linspace(a, b, m1)
uy = np.linspace(a, b, m2)

hx = ux[1] - ux[0]
hy = uy[1] - uy[0]

XX, YY = np.meshgrid(ux, uy, indexing="ij")
U = np.column_stack([XX.ravel(), YY.ravel()])

#boundary conditions

def interp_weights(x, U, h):
    """Vectorized: x can be scalar or array"""
    x = np.atleast_1d(x)
    W = np.zeros((len(x), len(U)))
    for i, xi in enumerate(x):
        j = np.searchsorted(U, xi) - 1
        j = np.clip(j, 0, len(U) - 2)
        t = (xi - U[j]) / (U[j+1] - U[j])
        W[i, j]   = 1 - t
        W[i, j+1] = t
    return W if len(x) > 1 else W[0]

def interp_weights_d1(x, U, h):
    """Vectorized: x can be scalar or array. Returns derivatives."""
    x = np.atleast_1d(x)
    W1 = np.zeros((len(x), len(U)))
    for i, xi in enumerate(x):
        j = np.searchsorted(U, xi)
        j = np.clip(j, 1, len(U) - 2)
        W1[i, j-1] =  1 / h
        W1[i, j+1] = -1 / h
    return W1 if len(x) > 1 else W1[0]

def interp_weights_d2(x, U, h):
    """Vectorized: x can be scalar or array. Returns second derivatives."""
    x = np.atleast_1d(x)
    W2 = np.zeros((len(x), len(U)))
    for i, xi in enumerate(x):
        j = np.searchsorted(U, xi)
        j = np.clip(j, 1, len(U) - 2)
        W2[i, j-1] =  1 / h**2
        W2[i, j]   = -2 / h**2
        W2[i, j+1] =  1 / h**2
    return W2 if len(x) > 1 else W2[0]



def kron_mv(Kx,Ky,v):
    V = v.reshape(m1, m2)
    KV = Kx @ V @ Ky.T
    return KV.ravel()

def rbf_kernel(X, l, sigma):
    sqdist_x = (X[:, 0][:, None] - X[:, 0][None, :]) ** 2
    sqdist_y = (X[:, 1][:, None] - X[:, 1][None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * (sqdist_x + sqdist_y) / l**2)

def build_K(l, sigma, sigma_n,v,kappa,h,axis = 'x'):
    Kuu = rbf_kernel(U, l, sigma) + sigma_n*np.eye(U.shape[0])
    if axis == 'x':
        Wd1 = interp_weights_d1(x, U[:, 0],h)
        Wd2 = interp_weights_d2(x, U[:, 0],h)
       
    else:
       
        Wd1 = interp_weights_d1(y, U[:, 1],h)
        Wd2 = interp_weights_d2(y, U[:, 1],h)
        
    W = v*Wd1 - kappa*Wd2
    K = W@Kuu@W.T

    K += sigma_n**2 * np.eye(K.shape[0])
    return K




def nlml(theta,hx,hy):
    l, sigma, log_sigma_n,v,kappa = theta
    sigma_n = np.exp(log_sigma_n)

    Kx = build_K(l, sigma, sigma_n,v,kappa,hx,axis = 'x')
    Ky = build_K(l, sigma, sigma_n,v,kappa,hy,axis = 'y')
    K = LinearOperator((N, N), matvec=lambda v: kron_mv(Kx,Ky,v))

    alpha, info = cg(K, yf, rtol=1e-8, maxiter=5000)

    alpha = solve(K, yf)
    #sign, logdet = slogdet(K)
    sign_x, logdet_x = slogdet(Kx)
    sIgn_y, logdet_y = slogdet(Ky)

    logdet = m2 * logdet_x + m1 * logdet_y
    return 0.5 * yf @ alpha + 0.5 * logdet

theta0 = np.array([0.2, 1.0, np.log(1e-2), 0.5, 0.5])  # [l, sigma, log sigma_n, v, kappa]

bounds = [
    (0.05, 1.0),     # lengthscale
    (0.1, 5.0),      # kernel variance
    (-10, 0),        # log noise std
    (0.0, 1.0),      # v
    (0.0, 1.0)       # kappa
]

def nlp(theta):
    return nlml(theta,hx,hy) - log_prior(theta)

res = minimize(
    nlp,
    theta0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 50}
)

l_opt, sigma_opt, log_sigma_n_opt, v_opt, kappa_opt = res.x
sigma_n_opt = np.exp(log_sigma_n_opt)

print("Optimized hyperparameters:")
print(f"l = {l_opt:.4f}, sigma = {sigma_opt:.4f}, sigma_n = {sigma_n_opt:.4e}")
K_uu = rbf_kernel(U, l_opt, sigma_opt)
K_uu += 1e-6 * np.eye(U.shape[0])   # jitter

Kx= build_K(l_opt, sigma_opt, sigma_n_opt,v_opt,kappa_opt,hx,axis = 'x')
Ky = build_K(l_opt, sigma_opt, sigma_n_opt,v_opt,kappa_opt,hy,axis = 'y')
K = kron_mv(Kx, Ky)
W = v_opt*interp_weights_d1(x, U[:, 0],hx) - kappa_opt*interp_weights_d2(x, U[:, 0],hx)
alpha, info = cg(K, yf, rtol=1e-8, maxiter=5000)
u_mean = K_uu @ W.T @ alpha

# Plot estimated and real solutions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Estimated solution
u_mean_grid = u_mean.reshape(m1, m2)
im1 = axes[0].contourf(XX, YY, u_mean_grid, levels=20, cmap='viridis')
axes[0].set_title('Estimated Solution')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# Real solution
yf_grid = f(XX, YY)
im2 = axes[1].contourf(XX, YY, yf_grid, levels=20, cmap='viridis')
axes[1].set_title('True Solution')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

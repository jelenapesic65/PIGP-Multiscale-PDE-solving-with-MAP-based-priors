import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize

import numpy as np

def u_exact(x, y, kappa, v1, v2):
    pi = np.pi
    
    alpha = v1 / (2 * kappa)
    beta  = v2 / (2 * kappa)
    
    lam = (v1**2 + v2**2) / (4 * kappa)
    
    denom = kappa * (2 * pi**2) + lam
    
    return (
        np.exp(alpha * x + beta * y)
        * np.sin(pi * x)
        * np.sin(pi * y)
        / denom
    )

def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

#def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

def rbf_kernel_1d(x, y, l, sigma):
    r = x-y
    return sigma**2 * np.exp(-0.5 * r**2 / l**2)

def rbf_kernel(X, Y, l, sigma):
    # X: (N,2), Y: (M,2)
    sqdist = (
        (X[:, None, 0] - Y[None, :, 0])**2 +
        (X[:, None, 1] - Y[None, :, 1])**2
    )
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

def interp_weights(x, U, h):
    W = np.zeros(len(U))
    j = np.searchsorted(U, x) - 1
    j = np.clip(j, 0, len(U) - 2)
    t = (x - U[j]) / (U[j+1] - U[j])
    W[j]   = 1 - t
    W[j+1] = t
    return W

def interp_weights_d1(x, U, h):
    W1 = np.zeros(len(U))
    j = np.searchsorted(U, x) - 1
    j = np.clip(j, 0, len(U) - 2)
    W1[j]   = -1 / h
    W1[j+1] =  1 / h
    return W1

def interp_weights_d2(x, U, h):
    W2 = np.zeros(len(U))
    j = np.searchsorted(U, x) - 1
    j = np.clip(j, 0, len(U) - 2)
    if j > 0:
        W2[j-1] = 1/h**2
    W2[j]     = -2 / h**2
    W2[j+1]   = 1 / h**2
    return W2

def build_K(u,x,c,l,sigma,sigma_n,v1,v2,kappa,with_bc=False):
    Kuu = rbf_kernel(u, u, l, sigma) 
    cx= c[0,:]
    cy = c[1,:]
    xx = x[0,:]
    yy = x[1,:]

    Wc  = np.vstack([interp_weights(xi, u, h) for xi in x[2:]])

    Wd1x = np.vstack([interp_weights_d1(xi, u, h) for xi in cx])
    Wd1y = np.vstack([interp_weights_d1(xi, u, h) for xi in cy])

    Wd2x  = np.vstack([interp_weights_d2(xi, u, h) for xi in cx])
    Wd2y  = np.vstack([interp_weights_d2(xi, u, h) for xi in cy])    

    Wpde = v1*Wd1x + v2*Wd1y - kappa*(Wd2x+Wd2y)

    if with_bc:
        # Boundary condition rows
        W_bc = np.zeros((2, m))
        W_bc[0, 0] = 1.0   # u(0)=0
        W_bc[1, -1] = 1.0  # u(1)=0
        W = np.vstack([W_bc, Wc, Wpde])
    else:
        W = np.vstack([Wc, Wpde])

    return Kuu,W#W@Kuu@W.T 

def build_K_star(u,x,c,x_star, l, sigma,sigma_n,v1,v2,kappa,with_bc=False):
    Kuu = rbf_kernel(u, u, l, sigma)
    #Wc = np.vstack([interp_weights(xi, u, h) for xi in c])
    #W = np.vstack([interp_weights(xi, u, h) for xi in x])
    
    W_star = np.vstack([interp_weights(xi, u, h) for xi in x_star])
  
  
    return Kuu, W_star

def log_prior(theta):
    l, sigma, log_sigma_n,v1,v2,kappa = theta

    log_l = np.log(l)
    log_sigma = np.log(sigma) #force positivity of hyperparameters

    # Hyperprior parameters
    #the bounds are made wrt to the fact that the domain is [0,1] and the expected behavior of the equation

    mu_l, std_l = np.log(0.3), 0.8 #multiplicative deviations
    mu_s, std_s = np.log(1.0), 0.5
    mu_n, std_n = np.log(1e-2), 1.0
    mu_v1, std_v1 = 0.5, 0.5
    mu_v2, std_v2 = 0.5, 0.5
    mu_k, std_k = 0.5, 0.5

    lp  = -0.5 * ((log_l - mu_l)/std_l)**2
    lp += -0.5 * ((log_sigma - mu_s)/std_s)**2
    lp += -0.5 * ((log_sigma_n - mu_n)/std_n)**2
    lp += -0.5 * ((v1 - mu_v1)/std_v1)**2
    lp += -0.5 * ((v2 - mu_v2)/std_v2)**2
    lp += -0.5 * ((kappa - mu_k)/std_k)**2

    return lp


def nlml(u,x,theta,with_bc=False):
    l, sigma, log_sigma_n,v1,v2,kappa = theta
    sigma_n = np.exp(log_sigma_n)
    
    Kuu, W = build_K(u,x,c,l,sigma,sigma_n,v1,v2,kappa,with_bc=with_bc)
    if not with_bc:
        x = x[2:]  # Remove boundary points
    K = W @ Kuu @ W.T + sigma_n**2 * np.eye(nxc)

    sign, logdet = slogdet(K)
   
    #check sign
    if sign <= 0:
        return np.inf
    #alpha, info = cg(K, y, rtol=1e-8, maxiter=5000)
    alpha = solve(K, y)
    return 0.5 * y.T @ alpha + 0.5 * logdet




a,b = 0.0, 5.0
n = 100
m = 70
nt = 30
nc = 50
nxc = n + nc    

xx= np.sort(np.random.uniform(a, b, n-2)) #n-2
xy = np.sort(np.random.uniform(a, b, n-2)) #n-2
#concatenate to get 2D points
x = np.vstack([xx, xy]) 

cx = np.sort(np.random.uniform(a, b, nc))
cy = np.sort(np.random.uniform(a, b, nc))
c = np.vstack([cx, cy])
ux = np.linspace(a, b, m)
uy = np.linspace(a, b, m)
Ux, Uy = np.meshgrid(ux, uy, indexing="ij")
u = np.column_stack([Ux.ravel(), Uy.ravel()])#t = np.random.uniform(a, b, nt)
x_starx = np.sort(np.random.uniform(a, b, 30))
x_stary = np.sort(np.random.uniform(a, b, 30))
x_star = np.vstack([x_starx, x_stary])
h = u[1] - u[0]
#idx_f = np.arange(0,n)
idx_u = np.arange(0,m)
#yt = f(x_star)
#yf = f(x) + np.random.normal(0, 0.001, size=len(x))
kappa_true = 0.6
v1_true = 0.3

v2_true = 0.2

#yf = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))
yt = u_exact(x_starx, x_stary, kappa=kappa_true, v1=v1_true, v2=v2_true) + np.random.normal(0, 0.001, size=len(x_star))
yu = u_exact(xx,xy, kappa=kappa_true, v1=v1_true, v2=v2_true) + np.random.normal(0, 0.001, size=len(x))

ybx =  np.array([0.0, 0.0])
yb = np.vstack([ybx, ybx])
#y = np.hstack([yu, yf])
xx = np.hstack([0.0,1.0, xx])  # add boundary points
xy = np.hstack([0.0,1.0, xy])  # add boundary points
x = np.vstack([xx, xy])

yf = f(cx,cy) + np.random.normal(0, 0.001, size=len(c))
y = np.hstack([yb, yu, yf])



l = 0.2
sigma = 0.3
sigma_n = 1e-2
v1 = 0.5
v2 = 0.5
kappa = 0.3

Kuu_prior = rbf_kernel(u, u, l, sigma)
prior_mean = np.zeros_like(u)
prior_std = np.sqrt(np.diag(Kuu_prior))

theta0 = np.array([l, sigma, np.log(sigma_n), v1,v2, kappa])

bounds = [
    (0.05, 2.0),     # lengthscale
    (0.1, 1.0),      # kernel variance
    (-10, 0),       # log noise variance
    (0.01, 1),    # v1
    (0.01, 1),    # v2
    (0.01, 1)     # kappa
]

def nlp(theta):
    return nlml(u,x,theta, with_bc=True) - log_prior(theta)

res = minimize(
    nlp,
    theta0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter":100}
)

l_opt, sigma_opt, log_sigma_n_opt, v1_opt, v2_opt, kappa_opt = res.x
sigma_n_opt = np.exp(log_sigma_n_opt)


#Overwrite the PDE parameters for test
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#kappa_opt = kappa_true
#v_opt = v_true
nxc = len(x) + len(c)

print("Optimized hyperparameters:")
print(f"l = {l_opt:.4f}, sigma = {sigma_opt:.4f}, sigma_n = {sigma_n_opt:.4e}, v1 = {v1_opt:.4f}, v2 = {v2_opt:.4f}, kappa = {kappa_opt:.4f}")

Kuu, W = build_K(u,x,c,l_opt, sigma_opt, sigma_n_opt, v1_opt, v2_opt, kappa_opt, with_bc=True)
K = W @ Kuu @ W.T +sigma_n_opt**2 * np.eye(nxc)
print("cond(K) =", np.linalg.cond(K))
Kuu,Wstar = build_K_star(u,x,c,x_star, l_opt, sigma_opt, sigma_n_opt, v1_opt, v2_opt, kappa_opt, with_bc=True)

alpha,info = cg(K,y,rtol=1e-8,maxiter=5000)
print("CG info:", info)
#posterior
Kuu = rbf_kernel(u, u,l_opt, sigma_opt)
#_mean = Kuu @ Wuz.T @ alpha
u_mean = Wstar @ Kuu @ W.T @ alpha
#Interior points only 

Kuu_int, W_int = build_K(u,x,c,l_opt, sigma_opt, sigma_n_opt, v1_opt, v2_opt, kappa_opt, with_bc=False)
K_int = W_int @ Kuu_int @ W_int.T + sigma_n_opt**2 * np.eye(nxc-2)
Kuu_int, W_star_int = build_K_star(u,x,c,x_star, l_opt, sigma_opt, sigma_n_opt, v1_opt, v2_opt, kappa_opt, with_bc=False)

alpha_int, info_int = cg(K_int, y[2:], rtol=1e-8, maxiter=5000)
print("CG info (internal only):", info_int)
u_mean_int = W_star_int @ Kuu_int @ W_int.T @ alpha_int
u_var = Wstar @ Kuu @ Wstar.T - Wstar @ Kuu @ W.T @ solve(K, W @ Kuu @ Wstar.T)


idx_t = np.argsort(x_star)
x_star_sorted = x_star[idx_t]

axx = np.linspace(a, b, 1000)
axy = np.linspace(a, b, 1000)
# prepare sorted predictions to match sorted x_star
u_mean_sorted = u_mean[idx_t]
u_mean_int_sorted = u_mean_int[idx_t]

plt.figure(figsize=(8, 5))

plt.plot(x_star_sorted, u_mean_sorted, label="PI-GP (KISS-GP, FD)")
plt.plot(x_star_sorted, u_mean_int_sorted, label="PI-GP (KISS-GP, FD, internal only)")
plt.scatter(x_star, yt, color='green', label="Test points (x_star)", zorder=1)
plt.scatter(x_star, u_mean_sorted, color='red', label="Predictions at x_star", zorder=1)
plt.scatter(x,y[:n], color='orange', label="Training points (x)", zorder=1)
plt.plot(axx, u_exact(axx, axy, kappa_true, v1_true, v2_true), '--', label="Exact solution (continuous)")
#plt.scatter(X_f, y_f, color='red', label="Collocation points (y_f)", zorder=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()


plt.figure(figsize=(10,6))
plt.fill_between(u, prior_mean-2*prior_std, prior_mean+2*prior_std, color='gray', alpha=0.3, label='Prior 2σ')
plt.fill_between(x_star_sorted, u_mean_sorted - 2*np.sqrt(u_var[idx_t]), u_mean_sorted + 2*np.sqrt(u_var[idx_t]), color='red', alpha=0.3, label='Posterior 2σ (with BCs)')
plt.plot(x_star_sorted, u_mean_int_sorted, 'b', label='Posterior (internal points only)')
plt.plot(x_star_sorted, u_mean_sorted, 'r', label='Posterior (internal + BCs)')
plt.plot(axx,  u_exact(axx, axy, kappa_true, v1_true, v2_true), '--', label='Exact solution (for reference)')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Prior vs Posterior with/without Boundary Conditions")
plt.show()

plt.figure(figsize=(10,6))
ls = np.linspace(0.05, 1.0, 10)
nlml_vals = [nlp([l_val, sigma, np.log(sigma_n), v1_opt, v2_opt, kappa_opt]) for l_val in ls]
plt.plot(ls, nlml_vals)
plt.show()
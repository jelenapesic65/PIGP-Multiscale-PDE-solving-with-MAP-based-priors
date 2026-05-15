import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg,LinearOperator
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import time


def u_exact(x, kappa, mu):
    import numpy as np
    pi = np.pi

    denom = pi**2 * kappa**2 + mu**2

    A = kappa / denom
    B = -mu / (np.pi * denom)

    # homogeneous part exponent
    r = mu / kappa

    #  C1, C2 using boundary conditions
    # u(0)=0
    # u(1)=0

    # At x=0:
    # C1 + C2 + B = 0

    # At x=1:
    # C1 + C2*exp(r) + A sin(pi) + B cos(pi) = 0
    # sin(pi)=0, cos(pi)=-1

    # => C1 + C2 e^r - B = 0

    e = np.exp(r)
    C2 = (2*B) / (e - 1)
    C1 = -B - C2

    return (
        C1
        + C2*np.exp(r*x)
        + A*np.sin(pi*x)
        + B*np.cos(pi*x)
    )



def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

def build_toeplitz_rbf(u, l, sigma):
    h = u[1] - u[0]
    m = len(u)
    dists = h * np.arange(m)
    col = sigma**2 * np.exp(-0.5 * (dists**2) / l**2)
    return col

def build_circulant_eigs(toeplitz_col):
    m = len(toeplitz_col)
    # circulant embedding
    c = np.concatenate([toeplitz_col,
                        toeplitz_col[-2:0:-1]])
    eigs = np.fft.fft(c)
    return eigs

def make_Kuu_mv_fft_operator(u, l, sigma):
    col = build_toeplitz_rbf(u, l, sigma)
    eigs = build_circulant_eigs(col)
    m = len(u)

    def Kuu_mv(v):
        v_pad = np.zeros(len(eigs), dtype=np.complex128)
        v_pad[:m] = v
        v_fft = np.fft.fft(v_pad)
        result = np.fft.ifft(eigs * v_fft).real
        return result[:m]

    return Kuu_mv


def f(x):
    return np.sin(np.pi * x)

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
    W2[j-1]   = 1 / h**2
    W2[j]     = -2 / h**2
    W2[j+1]   = 1 / h**2
    return W2

def build_W_sparse(x, u, h, deriv_order=0):
    rows, cols, vals = [], [], []
    for i, xi in enumerate(x):
        # compute interpolation row
        if deriv_order == 0:
            W_row = interp_weights(xi, u, h)
        elif deriv_order == 1:
            W_row = interp_weights_d1(xi, u, h)
        elif deriv_order == 2:
            W_row = interp_weights_d2(xi, u, h)
        else:
            raise ValueError("deriv_order must be 0,1,2")

        nz = np.nonzero(W_row)[0]  # only nonzeros
        for j in nz:
            rows.append(i)
            cols.append(j)
            vals.append(W_row[j])

    W_sparse = csr_matrix((vals, (rows, cols)), shape=(len(x), len(u)))
    return W_sparse


def build_K(u,x,c,l,sigma,sigma_n,v,kappa,with_bc=False):
    #Kuu_mv = rbf_kernel(u, u, l, sigma) 
    
    #Wc  = np.vstack([interp_weights(xi, u, h) for xi in x[2:]])

    #Wd1 = np.vstack([interp_weights_d1(xi, u, h) for xi in c])
    #Wd2 = np.vstack([interp_weights_d2(xi, u, h) for xi in c])
    Wc = build_W_sparse(x[2:], u, h, deriv_order=0)
    Wd1 = build_W_sparse(c, u, h, deriv_order=1)
    Wd2 = build_W_sparse(c, u, h, deriv_order=2)

    Wpde = v*Wd1 - kappa*Wd2

    if with_bc:
        # Boundary condition rows
        W_bc = np.zeros((2, m))
        W_bc[0, 0] = 1.0   # u(0)=0
        W_bc[1, -1] = 1.0  # u(1)=0
        W = vstack([W_bc, Wc, Wpde])
    else:
        W = vstack([Wc, Wpde])

    return W#W@Kuu_mv@W.T 

def build_K_star(u,x,c,x_star, l, sigma,sigma_n,v,kappa,with_bc=False):
    #W = np.vstack([interp_weights(xi, u, h) for xi in x])
    W_star = build_W_sparse(x_star, u, h, deriv_order=0)
  
    return W_star

def log_prior(theta):
    l, sigma, log_sigma_n,v,kappa = theta

    log_l = np.log(l)
    log_sigma = np.log(sigma) #force positivity of hyperparameters

    # Hyperprior parameters
    #the bounds are made wrt to the fact that the domain is [0,1] and the expected behavior of the equation

    mu_l, std_l = np.log(0.3), 0.8 #multiplicative deviations
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


def nlml(u,x,y,theta,with_bc=False):
    
    l, sigma, log_sigma_n, v, kappa = theta
    sigma_n = np.exp(log_sigma_n)
    
    W = build_K(u, x, c,l, sigma, sigma_n, v, kappa, with_bc=with_bc)
    n_system = W.shape[0]
    def K_mv(v):
        return W @ (Kuu_mv @ (W.T @ v)) + sigma_n**2 * v

    K_linop = LinearOperator((n_system, n_system), matvec=K_mv)
    
    alpha, info = cg(K_linop, y, rtol=1e-8, maxiter=5000)
    
    n_trace = min(50, max(10, n_system // 100))    
    logdet_est = 0.0
    for _ in range(n_trace):
        z = np.random.choice([-1.0, 1.0], size=n_system)  # Rademacher vector
        v, info_z = cg(K_linop, z, rtol=1e-8, maxiter=5000)
        logdet_est += z @ v
    logdet_est /= n_trace
    logdet_est = np.log(logdet_est + 1e-12)  # avoid log(0)
   
    #K = W @ Kuu_mv @ W.T + sigma_n**2 * np.eye(nxc)

    #sign, logdet_est = slogdet(K)

    nlml_val = 0.5 * y.T @ alpha + 0.5 * logdet_est
    return nlml_val

a,b = 0.0, 5.0
n = 100
m = 70
nt = 30
nc = 50
nxc = n+nc

x= np.sort(np.random.uniform(a, b, n-2)) #n-2
c = np.sort(np.random.uniform(a, b, nc))
u = np.linspace(a, b, m)
#t = np.random.uniform(a, b, nt)
x_star = np.sort(np.random.uniform(a, b, 30))
h = u[1] - u[0]
#idx_f = np.arange(0,n)
idx_u = np.arange(0,m)
#yt = f(x_star)
#yf = f(x) + np.random.normal(0, 0.001, size=len(x))
kappa_true = 0.6
v_true = 0.3


#yf = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))
yt = u_exact(x_star, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x_star))
yu = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))

yb =  np.array([0.0, 0.0])
#y = np.hstack([yu, yf])
x = np.hstack([0.0,1.0, x])  # add boundary points
yf = f(c) + np.random.normal(0, 0.001, size=len(c))
y = np.hstack([yb, yu, yf])

l = 0.2
sigma = 0.3
sigma_n = 1e-2
v = 0.5
kappa = 0.3


Kuu_mv_prior = rbf_kernel(u, u, l, sigma)
Kuu_mv = Kuu_mv_prior
prior_mean = np.zeros_like(u)
prior_std = np.sqrt(np.diag(Kuu_mv_prior))

theta0 = np.array([l, sigma, np.log(sigma_n), v, kappa])

bounds = [
    (0.05, 2.0),     # lengthscale
    (0.1, 5.0),      # kernel variance
    (-10, 0),       # log noise variance
    (0.01, 1.0),    # v
    (0.01, 1.0)     # kappa
]

def nlp(theta):
    return nlml(u,x,y,theta, with_bc=True) - log_prior(theta)

start = time.time()

res = minimize(
    nlp,
    theta0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter":50}
)

l_opt, sigma_opt, log_sigma_n_opt, v_opt, kappa_opt = res.x
sigma_n_opt = np.exp(log_sigma_n_opt)

#Overwrite the PDE parameters for test
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#kappa_opt = kappa_true
#v_opt = v_true
nxc = len(x) + len(c)


print("Optimized hyperparameters:")
print(f"l = {l_opt:.4f}, sigma = {sigma_opt:.4f}, sigma_n = {sigma_n_opt:.4e}, v = {v_opt:.4f}, kappa = {kappa_opt:.4f}")

W = build_K(u,x,c,l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt, with_bc=True)
#K = W @ Kuu_mv @ W.T +sigma_n_opt**2 * np.eye(n)

def K_mv(v):
    return W @ (Kuu_mv @ (W.T @ v)) + sigma_n_opt**2 * v
 

n_system = W.shape[0]
K = LinearOperator((n_system, n_system), matvec=K_mv)

Wstar = build_K_star(u,x,c,x_star, l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt, with_bc=True)

alpha,info = cg(K,y,rtol=1e-8,maxiter=5000)
#print("cond(K) =", np.linalg.cond(K))

print("CG info:", info)
#posterior

#u_mean = Wstar @ Kuu_mv @ W.T @ alpha #here the K is fully formed and that defeates the purpose of W being sparse
u_mean = Wstar @ (Kuu_mv @ (W.T @ alpha))
#Interior points only
end = time.time()

print(f"Total time: {end - start:.2f} seconds")
Kuu_mv_int = Kuu_mv
W_int = build_K(u,x,c,l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt, with_bc=False)
#K_int = W_int @ Kuu_mv_int @ W_int.T + sigma_n_opt**2 * np.eye(nxc-2)
K_int = LinearOperator((nxc-2, nxc-2), matvec=lambda v: W_int @ (Kuu_mv_int @ (W_int.T @ v)) + sigma_n_opt**2 * v)



W_star_int = build_K_star(u,x,c,x_star, l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt, with_bc=False)

alpha_int, info_int = cg(K_int, y[2:], rtol=1e-8, maxiter=5000)
print("CG info (internal only):", info_int)
u_mean_int = W_star_int @ (Kuu_mv_int @ (W_int.T @ alpha_int))

kappa = 0.4
mu = 0.2

idx_t = np.argsort(x_star)
x_star_sorted = x_star[idx_t]

ax = np.linspace(a, b, 1000)
# prepare sorted predictions to match sorted x_star
u_mean_sorted = u_mean[idx_t]
u_mean_int_sorted = u_mean_int[idx_t]

plt.figure(figsize=(8, 5))

plt.plot(x_star_sorted, u_mean_sorted, label="PI-GP (KISS-GP, FD)")
plt.plot(x_star_sorted, u_mean_int_sorted, label="PI-GP (KISS-GP, FD, internal only)")
# scatter doesn't accept line-style strings; use plot for dashed black line
plt.plot(ax, u_exact(ax, kappa_true, v_true), '--', label="Exact solution (continuous)")
#plt.scatter(X_f, y_f, color='red', label="Collocation points (y_f)", zorder=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()


plt.figure(figsize=(10,6))
plt.fill_between(u, prior_mean-2*prior_std, prior_mean+2*prior_std, color='gray', alpha=0.3, label='Prior 2σ')
plt.plot(x_star_sorted, u_mean_int_sorted, 'b', label='Posterior (internal points only)')
plt.plot(x_star_sorted, u_mean_sorted, 'r', label='Posterior (internal + BCs)')

plt.plot(u,  u_exact(u, kappa_true, v_true), '--', label='Exact solution (for reference)')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Prior vs Posterior with/without Boundary Conditions")
plt.show()

plt.figure(figsize=(10,6))
ls = np.linspace(0.05, 1.0, 10)
nlml_vals = [nlp([l_val, sigma, np.log(sigma_n), v, kappa]) for l_val in ls]
plt.plot(ls, nlml_vals)
plt.show()
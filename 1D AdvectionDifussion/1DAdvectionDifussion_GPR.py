from networkx import sigma
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize

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

def build_K_star(x_star, x, l, sigma, v, kappa):
    xb = x[:2]
    xf = x[2:]

    kbb, *_ = kernel_derivatives(x_star, xb, l, sigma)

    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x_star, xf, l, sigma)
    Kuf = v * k_xp - kappa * k_xpxp

    return np.hstack([kbb, Kuf])

def kernel_derivatives(x,y,l,sigma):
    r = x[:, None] - y[None, :]
    k = sigma**2 * np.exp(-0.5 * r**2 / l**2)

    l2 = l**2
    l4 = l**4
    l6 = l**6
    l8 = l**8

    k_x  = -(r/l2) * k
    k_xp = +(r/l2) * k

    k_xx  = (r**2/l4 - 1/l2) * k
    k_xpxp = k_xx
    k_xxp = (1/l2 - r**2/l4) * k

    k_xxxp = (-3*r/l4 + r**3/l6) * k # -3/l4 r+1/l6 r^3 
    #k_xxpxp = (3*r**2/l6 - 1/l4) * k # 3/l4r-r^3/l6
    k_xxpxp = (3*r/l4 - r**3/l6) * k # 3/l4r-r^3/l6

    k_xxxpxp = (3/l4 - 6*r**2/l6 + r**4/l8) * k
    
    #if (k_xxxp.shape[0] == k_xxxp.shape[1]):
    #    print('Symmetry check (k_xxxp - k_xxxp.T):', np.max(np.abs(k_xxxp - k_xxxp.T)))
    #    print('Symmetry check (k_xxpxp - k_xxpxp.T):', np.max(np.abs(k_xxpxp - k_xxpxp.T)))
    #    print('Symmetry check (k_xxxpxp - k_xxxpxp.T):', np.max(np.abs(k_xxxpxp - k_xxxpxp.T)))

    return k, k_x, k_xp, k_xx, k_xpxp, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp



def rbf_kernel(x, y, l=0.2, sigma=1.0):
    sqdist = (x[:, None] - y[None, :]) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / l**2)

def f(x):
    return np.sin(np.pi * x)


def build_K(x,l,sigma,sigma_n,v,kappa,with_bc=False):
    xb = x[:2]  # boundary points
    xf = x[2:]  # interior points
    nb = len(xb)  
    nf = len(xf)
    x = np.hstack([xb, xf])  # ensure x is ordered with boundaries first, check different ordering 
    K = np.zeros((len(x), len(x)))


    kbb,*_ = kernel_derivatives(xb, xb, l, sigma)
    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(xb, xf, l, sigma)

    Kbf = v * k_xp - kappa * k_xpxp
    K[:nb,:nb]  = kbb
    K[:nb, nb:] = Kbf

    
    kfb, k_x, _, k_xx, *_ = kernel_derivatives(xf, xb, l, sigma)

    Kfb = v * k_x - kappa * k_xx
    K[nb:, :nb] = Kfb

    (_, _, _, _, _, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp) = kernel_derivatives(xf, xf, l, sigma)

    #Kff = v**2 * k_xxp - v*kappa * k_xxxp - v*kappa * k_xxpxp + kappa**2 * k_xxxpxp
    # kxxxp = - kxxpxp
    Kff = v**2 * k_xxp + kappa**2 * k_xxxpxp
    Kff += sigma_n**2 * np.eye(nf)

    K[nb:, nb:] = Kff

    if not with_bc:
        return Kff
    
    return K

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


def nlml(x,theta,with_bc=False):
    l, sigma, log_sigma_n,v,kappa = theta
    sigma_n = np.exp(log_sigma_n)
    
    K = build_K(x,l,sigma,sigma_n,v,kappa,with_bc)
    
    sign, logdet = slogdet(K)

    alpha, info = cg(K, y, rtol=1e-8, maxiter=5000)
    return 0.5 * y.T @ alpha + 0.5 * logdet

a,b = 0.0, 1.0
n = 1000
m = 500
nt = 300



x= np.random.uniform(a, b, n-2)
x = np.sort(x)  
#y = np.linspace(a, b, n-2)  # use linspace grid for better conditioning
t = np.random.uniform(a, b, nt)
x_star = np.random.uniform(a, b, 30)
#idx_f = np.arange(0,n)

kappa_true = 0.6
v_true = 0.3

yf = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))
yu =  np.array([0.0, 0.0])
y = np.hstack([yu, yf])
x = np.hstack([0.0,1.0, x])  # add boundary points
l = 0.2
sigma = 0.3
sigma_n = 1e-2
v = 0.2
kappa = 0.7

Kxx_prior = rbf_kernel(x, x, l, sigma)
prior_mean = np.zeros_like(x)
prior_std = np.sqrt(np.diag(Kxx_prior))

theta0 = np.array([l, sigma, np.log(sigma_n), v, kappa])

bounds = [
    (0.05, 2.0),     # lengthscale
    (0.1, 5.0),      # kernel variance
    (-10, 0),       # log noise variance
    (0.01, 1.0),    # v
    (0.01, 1.0)     # kappa
]

def nlp(theta):
    return nlml(x,theta, with_bc=True) - log_prior(theta)

res = minimize(
    nlp,
    theta0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter":50}
)

l_opt, sigma_opt, log_sigma_n_opt, v_opt, kappa_opt = res.x
sigma_n_opt = np.exp(log_sigma_n_opt)

print("Optimized hyperparameters:")
print(f"l = {l_opt:.4f}, sigma = {sigma_opt:.4f}, sigma_n = {sigma_n_opt:.4e}, v = {v_opt:.4f}, kappa = {kappa_opt:.4f}")

Kxx= build_K(x,l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt, with_bc=True)
Kxx = Kxx + 1e-10 * np.eye(len(x)) #jitter for numerical stability
print('Max deviation from symmetry:', np.max(np.abs(Kxx - Kxx.T)))

print("cond(K) =", np.linalg.cond(Kxx))
alpha,info = cg(Kxx,y,rtol=1e-8,maxiter=5000)
print("CG info:", info)
#posterior
#Kxx = rbf_kernel(x, x,l_opt, sigma_opt)
Kuz = build_K_star(x_star, x, l_opt, sigma_opt, v_opt, kappa_opt)
u_mean = Kuz @ alpha

#Interior points only 

Kxx_int = build_K(x,l_opt, sigma_opt, sigma_n_opt, v_opt, kappa_opt, with_bc=False)
Kxx_int = Kxx_int + 1e-10 * np.eye(len(x)-2) #jitter for numerical stability
alpha_int, info_int = cg(Kxx_int, yf, rtol=1e-8, maxiter=5000)
print("CG info (internal only):", info_int)
Kuz = build_K_star(x_star, x, l_opt, sigma_opt, v_opt, kappa_opt)
u_mean_int = Kuz[:, 2:] @ alpha_int
idx = np.argsort(x_star)

plt.figure(figsize=(8, 5))


plt.plot(x_star[idx], u_mean[idx], label="PI-GP (KISS-GP, FD)")
plt.plot(x, u_exact(x, kappa_true, v_true), '--', label="Exact solution")
#plt.scatter(X_f, y_f, color='red', label="Collocation points (y_f)", zorder=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()


plt.figure(figsize=(10,6))
plt.fill_between(x, prior_mean-2*prior_std, prior_mean+2*prior_std, color='gray', alpha=0.3, label='Prior 2σ')
plt.plot(x_star[idx], u_mean_int[idx], 'b', label='Posterior (internal points only)')
plt.plot(x_star[idx], u_mean[idx], 'r', label='Posterior (internal + BCs)')
plt.plot(x,  u_exact(x, kappa_true, v_true), '--', label='Exact solution (for reference)')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Prior vs Posterior with/without Boundary Conditions")
plt.show()

plt.figure(figsize=(10,6))
ls = np.linspace(0.05, 8.0, 15)
nlml_vals = [nlp([l_val, sigma, np.log(sigma_n), v, kappa]) for l_val in ls]
plt.plot(ls, nlml_vals)
plt.title("Negative Log Marginal Likelihood vs Lengthscale")
plt.show()

'''

l_vals = np.linspace(0.5*l_opt, 8*l_opt, 100)
k_vals = np.linspace(0.5*kappa_opt, 8*kappa_opt, 100)

L, K = np.meshgrid(l_vals, k_vals)
Z = np.zeros_like(L)

for i in range(len(l_vals)):
    for j in range(len(k_vals)):
        theta = [L[j,i], sigma_opt, np.log(sigma_n_opt), v_opt, K[j,i]]
        Z[j,i] = nlp(theta)

Z -= np.min(Z)

plt.figure(figsize=(8,6))
plt.contourf(L, K, Z, levels=30)
plt.scatter(l_opt, kappa_opt, color='red', s=80)
plt.xlabel("l")
plt.ylabel("kappa")
plt.colorbar(label="NLML - min")
plt.title("Interaction: l vs κ")
plt.show()
'''
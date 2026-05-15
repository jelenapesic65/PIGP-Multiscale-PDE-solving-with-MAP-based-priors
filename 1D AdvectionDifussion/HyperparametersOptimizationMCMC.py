# --- Full Bayesian integration on a discretized grid ---
# Assumes: x, c, y, x_star, nlp, log_prior, build_K, build_K_star are defined. :contentReference[oaicite:2]{index=2}
import arviz as az
import pymc as pm
import pytensor.tensor as tt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from numpy.linalg import slogdet, solve
from scipy.optimize import minimize
assert np.__version__ < "2", "numpy version must be < 2"

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

def build_K_star(x_star, x, c, l, sigma, v, kappa):
    xb = x[:2]
    xu = x[2:]
    xf = c 
    x = np.hstack([xb, xu])  # ensure x is ordered with boundaries first, check different ordering
    kbb, *_ = kernel_derivatives(x_star, x , l, sigma)

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
def nlp(theta):
    return nlml(x,theta, with_bc=True) - log_prior(theta)

def build_K(x,c,l,sigma,sigma_n,v,kappa,with_bc=False):
    
    xb = x[:2]  # boundary points
    xu = x[2:]  # interior points
    xf = c
    nf = len(xu)
    x = np.hstack([xb, xu])  # ensure x is ordered with boundaries first, check different ordering 
    nb = len(x)  

    nxc = len(x) + len(c)
    K = np.zeros((nxc, nxc))


    kbb,*_ = kernel_derivatives(x, x, l, sigma)
    kbf, _, k_xp, _, k_xpxp, *_ = kernel_derivatives(x, xf, l, sigma)

    Kbf = v * k_xp - kappa * k_xpxp
    K[:nb,:nb]  = kbb
    K[:nb, nb:] = Kbf

    
    kfb, k_x, _, k_xx, *_ = kernel_derivatives(xf, x, l, sigma)

    Kfb = v * k_x - kappa * k_xx
    K[nb:, :nb] = Kfb

    (_, _, _, _, _, k_xxp, k_xxxp, k_xxpxp, k_xxxpxp) = kernel_derivatives(xf, xf, l, sigma)

    #Kff = v**2 * k_xxp - v*kappa * k_xxxp - v*kappa * k_xxpxp + kappa**2 * k_xxxpxp
    # kxxxp = - kxxpxp
    Kff = v**2 * k_xxp + kappa**2 * k_xxxpxp
    Kff += sigma_n**2 * np.eye(len(c))

    K[nb:, nb:] = Kff

    if not with_bc:
        return K[2:,2:]
    
    return K

def nlml(x,theta,with_bc=False):
    l, sigma, log_sigma_n,v,kappa = theta
    sigma_n = np.exp(log_sigma_n)
    
    K = build_K(x,c,l,sigma,sigma_n,v,kappa,with_bc) 
    K += 1e-6 * np.eye(K.shape[0]) 
    sign, logdet = slogdet(K)

    #alpha, info = cg(K, y, rtol=1e-8, maxiter=5000)
    alpha = solve(K, y)
    return 0.5 * y.T @ alpha + 0.5 * logdet

a,b = 0.0, 5.0
n = 100
m = 70
nt = 70
nc = 10
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


yt = u_exact(x_star, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x_star))
yu = u_exact(x, kappa=kappa_true, mu=v_true) + np.random.normal(0, 0.001, size=len(x))

yb =  np.array([0.0, 0.0])
#y = np.hstack([yu, yf])
x = np.hstack([a,b, x])  # add boundary points
yf = f(c) + np.random.normal(0, 0.1, size=len(c))
y = np.hstack([yb, yu, yf])


l = 0.2
sigma = 0.3
sigma_opt = 0.6
sigma_n_opt = 1e-3
v_opt = 0.5
sigma_n = 1e-2
v = 0.8
kappa = 0.2
x_data = x.copy()
y_data = y.copy()
c_data = c.copy()
x_star_data = x_star.copy()

# Number of NUTS samples
n_samples = 2000
n_tune = 1000
Kxx_prior = rbf_kernel(x, x, l, sigma)
prior_mean = np.zeros_like(x)
prior_std = np.sqrt(np.diag(Kxx_prior))

with pm.Model() as gp_model:

    # Hyperpriors – use unconstrained variables where appropriate
    log_l      = pm.Normal("log_l", mu=np.log(0.3), sigma=1.0)
    log_sigma  = pm.Normal("log_sigma", mu=np.log(1.0), sigma=1.0)
    log_sigma_n = pm.Normal("log_sigma_n", mu=np.log(1e-2), sigma=1.0)

    v       = pm.Normal("v", mu=0.5, sigma=0.5)
    kappa   = pm.Normal("kappa", mu=0.5, sigma=0.5)

    # Stack hyperparameters
    theta = tt.stack([tt.exp(log_l),
                      tt.exp(log_sigma),
                      log_sigma_n,
                      v,
                      kappa])

    # Custom density for GP posterior (logp = −nlp)
    def _logp(theta_tt):

        # Convert PyTensor → numpy
        theta_val = theta_tt.eval()  # numeric values
        print(theta_val)
        # Compute negative log posterior
        val = -nlp(theta_val)

        return val

    # Use PyMC DensityDist for custom logp
    l, sigma, log_sigma_n, v, kappa = theta
    l = float(l)
    sigma = float(sigma)
    log_sigma_n = float(log_sigma_n)
    v = float(v)
    kappa = float(kappa)
    theta_val = [l, sigma, log_sigma_n, v, kappa]
    pm.DensityDist("likelihood", logp=_logp, observed= theta_val)

    # Run NUTS
    trace = pm.sample(
        draws=n_samples,
        tune=n_tune,
        chains=4,
        target_accept=0.9,
        return_inferencedata=True
    )

posterior_samples = trace.posterior

# Flatten samples
flat_samples = az.extract(trace, var_names=["log_l","log_sigma","log_sigma_n","v","kappa"])

pred_samples = []

for i in range(flat_samples["log_l"].shape[0]):
    log_l_s, log_sigma_s, log_sigma_n_s, v_s, kappa_s = \
        flat_samples["log_l"][i], flat_samples["log_sigma"][i], \
        flat_samples["log_sigma_n"][i], flat_samples["v"][i], \
        flat_samples["kappa"][i]

    theta_sample = np.array([
        np.exp(log_l_s),
        np.exp(log_sigma_s),
        log_sigma_n_s,
        v_s,
        kappa_s
    ])

    l_s, sigma_s, log_sigma_n_s, v_s, kappa_s = theta_sample
    sigma_n_s = np.exp(log_sigma_n_s)

    Kxx = build_K(x_data, c_data, l_s, sigma_s, sigma_n_s, v_s, kappa_s, with_bc=True)
    alpha = np.linalg.solve(Kxx, y_data)

    Kuz = build_K_star(x_star_data, x_data, c_data, l_s, sigma_s, v_s, kappa_s)

    pred_mean = Kuz @ alpha
    pred_samples.append(pred_mean)

pred_samples = np.stack(pred_samples)

post_mean = pred_samples.mean(axis=0)
post_std  = pred_samples.std(axis=0)

plt.figure(figsize=(10,6))
plt.plot(x_star_data, post_mean, label="Posterior Mean (NUTS)")
plt.fill_between(x_star_data,
                 post_mean - 2*post_std,
                 post_mean + 2*post_std,
                 alpha=0.3, label="±2 std")
plt.scatter(x_data, y_data[:len(x_data)], c="red", label="Obs")
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg

#linear 
def interp_weights(x, U): 
    W = np.zeros(len(U)) 
    j = np.searchsorted(U, x) - 1 
    j = np.clip(j, 0, len(U) - 2) 
    t = (x - U[j]) / (U[j+1] - U[j]) 
    W[j] = 1 - t
    W[j+1] = t 
    return W 
def interp_weights_d2(x, U): 
    W2 = np.zeros(len(U)) 
    j = np.searchsorted(U, x) 
    j = np.clip(j, 1, len(U) - 2) 
    W2[j-1] = 1 / h**2 
    W2[j] = -2 / h**2 
    W2[j+1] = 1 / h**2 
    return W2 
#cubic 
def interp_weights_cubic(x, U):
     W = np.zeros(len(U)) 
     j = np.searchsorted(U, x) - 1 
     j = np.clip(j, 1, len(U) - 3) 
     h = U[1] - U[0] 
     t = (x - U[j]) / h 
     W[j-1] = (-t**3 + 2*t**2 - t) / 6 
     W[j] = (3*t**3 - 5*t**2 + 2) / 6 
     W[j+1] = (-3*t**3 + 4*t**2 + t) / 6 
     W[j+2] = (t**3 - t**2) / 6 
     return W 
def interp_weights_d2_cubic(x, U): 
    W2 = np.zeros(len(U)) 
    j = np.searchsorted(U, x) - 1 
    j = np.clip(j, 1, len(U) - 3) 
    h = U[1] - U[0] 
    t= (x - U[j]) / h 
    W2[j-1] = (t - 1) / h**2 
    W2[j] = (-6*t + 3) / h**2 
    W2[j+1] = (6*t + 3) / h**2 
    W2[j+2] = (-t - 2) / h**2 
    return W2 
def f(x): 
    return np.sin(np.pi * x) 

a, b = 0.0, 1.0 
m = 20 
n_f = 20 
X_f = np.linspace(a, b, n_f)[1:-1] 
U = np.linspace(a, b, m) 
h = U[1] - U[0]
Wf_d2 = np.vstack([interp_weights_d2(x, U) for x in X_f]) 
# Interior collocation indices 
idx_f = np.arange(1, m - 1) 
# Boundary indices 
idx_u = np.array([0, m - 1]) 

def rbf_kernel(x, y, l=0.2, sigma=1.0):
     sqdist = (x[:, None] - y[None, :]) ** 2 
     return sigma**2 * np.exp(-0.5 * sqdist / l**2) 

K_uu = rbf_kernel(U, U) 
K_uu += 1e-6 * np.eye(m) # jitter

 # Boundary observation operator 
W_u = np.zeros((2, m)) 
W_u[0, 0] = 1.0 
W_u[1, -1] = 1.0 
W = np.vstack([W_u, Wf_d2]) #final operator 
# PDE RHS 
y_f = f(X_f) + np.random.normal(0, 0.1, size=len(X_f)) 
# Boundary values
y_u = np.array([0.0, 0.0]) # Stack operators 
#A = np.vstack([W_u, L_f.toarray()])
y = np.hstack([y_u, y_f]) 
K = np.zeros((len(y), len(y))) # Build covariance matrix K 
K[0:2, 0:2] = W_u @ K_uu @ W_u.T 
K[0:2, 2:] = W_u @ K_uu @ Wf_d2.T 
K[2:, 0:2] = Wf_d2 @ K_uu @ W_u.T 
K[2:, 2:] = Wf_d2 @ K_uu @ Wf_d2.T 
# Solve for alpha: 
alpha = (K + 1e-6 * np.eye(len(y))) @ y 
alpha, info = cg(K + 1e-6 * np.eye(len(y)), y, rtol=1e-8, maxiter=5000) 
u_mean = K_uu @ W.T @ alpha 
#u_mean = K_uu @ A.T @ alpha 
#AK = A @ K_uu #M = AK @ A.T + 1e-6 * np.eye(len(y)) #alpha, _ = cg(M, y)
#  # Posterior mean on inducing grid 
plt.figure(figsize=(8, 5)) 
plt.plot(U, u_mean, label="PI-GP (KISS-GP, FD)") 
plt.plot(U, -np.sin(np.pi * U) / (np.pi**2), '--', label="Exact solution") 
plt.scatter(X_f, -y_f / (np.pi**2), color='red', label="Collocation points (y_f)", zorder=1) 
plt.scatter(U[idx_u], y_u, color='green', marker='s', label="Boundary points", zorder=1) 
plt.xlabel("x") 
plt.ylabel("u(x)") 
plt.legend() 
plt.show() 

X_test = np.linspace(0, 1, 200)

W_star = np.vstack([interp_weights(x, U) for x in X_test]) # only interpolation without derivatives because we are predicting the function u itself 

#K_star_A = W_test @ K_uu @ A.T
#mu_star  = K_star_A @ alpha
K_star_x = W_star @ K_uu @ W.T

u_star = K_star_x @ alpha
u_true = -(1 / np.pi**2) * np.sin(np.pi * X_test)

K_star_star = W_star @ K_uu @ W_star.T

# v = np.linalg.solve(K_star_x, W_star @ K_uu @ W.T)
v = np.linalg.solve(K + 1e-6 * np.eye(len(y)), K_star_x.T)
Sigma_star = K_star_star - K_star_x @ v


plt.figure()
plt.plot(X_test, u_star, label="PI-GP prediction")
plt.plot(X_test, u_true, "--", label="True solution")
plt.scatter(X_f, -np.sin(np.pi * X_f) / (np.pi**2), color='red', label="Collocation points (y_f)", zorder=1)
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.show()
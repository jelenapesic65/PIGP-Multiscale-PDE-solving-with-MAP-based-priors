import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

np.random.seed(42)

# Grid setup
Nx, Nt = 50, 50
Lx, T = 8.0, 4.0
dx = Lx/(Nx-1)
dt = T/(Nt-1)

x_grid = np.linspace(0, Lx, Nx)
t_grid = np.linspace(0, T, Nt)
XX, TT = np.meshgrid(x_grid, t_grid)

# True parameters
p1_true, p2_true = 0.6, 0.08
sigma_obs = 0.02
sigma2 = sigma_obs**2

# True forcing field
f_true = (
    2.5*np.exp(-((XX-2.0)**2)/0.4 - ((TT-1.0)**2)/0.3)
    + 1.8*np.exp(-((XX-5.5)**2)/0.5 - ((TT-2.5)**2)/0.4)
)

# Build system matrix
def build_A(p1, p2):
    A = np.eye(Nx)

    # Advection
    for j in range(1, Nx):
        A[j, j]   -= dt*p1/dx
        A[j, j-1] += dt*p1/dx

    A[0, 0] -= dt*p1/dx
    A[0, 1] += dt*p1/dx

    # Diffusion
    for j in range(1, Nx-1):
        A[j, j-1] += dt*p2/dx**2
        A[j, j]   -= 2*dt*p2/dx**2
        A[j, j+1] += dt*p2/dx**2

    A[0, 0]     -= dt*p2/dx**2
    A[0, 1]     += dt*p2/dx**2
    A[-1, -1]   -= dt*p2/dx**2
    A[-1, -2]   += dt*p2/dx**2

    return A

# Forward solver
def forward_solve(f_field, p1, p2):
    A = build_A(p1, p2)
    u = np.zeros((Nt, Nx))
    for n in range(Nt-1):
        u[n+1] = A @ u[n] + dt * f_field[n]
    return u

# Adjoint solver
def adjoint_solve(p1, p2, xi, ti):
    A = build_A(p1, p2)
    w = np.zeros((Nt, Nx))
    if ti == 0:
        return w
    w[ti-1, xi] = 1.0
    for n in range(ti-1, 0, -1):
        w[n-1] = A.T @ w[n]
    return w

# Generate true data
u_true = forward_solve(f_true, p1_true, p2_true)

# Coarse parametrization of f
Nfx, Nft = 15, 15
xf = np.linspace(0, Lx, Nfx)
tf = np.linspace(0, T, Nft)
XXf, TTf = np.meshgrid(xf, tf)

M = Nfx * Nft  # number of parameters

def q_to_f(q_vec):
    Q = q_vec.reshape(Nft, Nfx)
    interp = RegularGridInterpolator(
        (tf, xf), Q, method='linear',
        bounds_error=False, fill_value=0
    )
    pts = np.stack([TT.ravel(), XX.ravel()], axis=1)
    return interp(pts).reshape(Nt, Nx)

# Build interpolation matrix B
B = np.zeros((Nt*Nx, M))
for m in range(M):
    e_m = np.zeros(M)
    e_m[m] = 1.0
    B[:, m] = q_to_f(e_m).ravel()

print(f"B shape: {B.shape}, B range: [{B.min():.3f},{B.max():.3f}]")

q_ls = np.linalg.lstsq(B, f_true.ravel(), rcond=None)[0]
f_approx = (B @ q_ls).reshape(Nt, Nx)

print(f"Bilinear interp MSE: {np.mean((f_approx - f_true)**2):.5f}")
print(f"||q_ls||: {np.linalg.norm(q_ls):.4f}")
print(f"q_ls range: [{q_ls.min():.3f},{q_ls.max():.3f}]")
print(f"f_true range: [{f_true.min():.3f},{f_true.max():.3f}]")

n_sx, n_st = 10, 10
sx_idx = np.linspace(2, Nx-3, n_sx, dtype=int)
st_idx = np.linspace(3, Nt-3, n_st, dtype=int)

obs_xi, obs_ti = np.meshgrid(sx_idx, st_idx)
obs_xi = obs_xi.ravel()
obs_ti = obs_ti.ravel()

n_obs = len(obs_xi)
z = u_true[obs_ti, obs_xi] + sigma_obs * np.random.randn(n_obs)

Phi = np.zeros((n_obs, M))
for i in range(n_obs):
    w = adjoint_solve(p1_true, p2_true, obs_xi[i], obs_ti[i])
    Phi[i] = dt * (w.ravel() @ B)

print(f"\nPhi range: [{Phi.min():.4f},{Phi.max():.4f}]")

tau2 = (q_ls @ q_ls) / M
print(f"tau2: {tau2:.4f}")

# Posterior
A_post = (1/sigma2)*Phi.T @ Phi + (1/tau2)*np.eye(M)
Sn = np.linalg.inv(A_post)
mu_n = (1/sigma2) * Sn @ Phi.T @ z

f_hat = (B @ mu_n).reshape(Nt, Nx)

print(f"\nf_hat range: [{f_hat.min():.3f},{f_hat.max():.3f}]")
print(f"f_true range: [{f_true.min():.3f},{f_true.max():.3f}]")
print(f"Corr(f_hat,f_true): {np.corrcoef(f_hat.ravel(),f_true.ravel())[0,1]:.4f}")
print(f"MSE(f_hat,f_true): {np.mean((f_hat-f_true)**2):.5f}")

# Q landscape
def build_Phi_post(p1, p2):
    Ph = np.zeros((n_obs, M))
    for i in range(n_obs):
        w = adjoint_solve(p1, p2, obs_xi[i], obs_ti[i])
        Ph[i] = dt * (w.ravel() @ B)

    A_ = (1/sigma2)*Ph.T @ Ph + (1/tau2)*np.eye(M)
    Sn_ = np.linalg.inv(A_)
    mn_ = (1/sigma2) * Sn_ @ Ph.T @ z
    return Ph, mn_, Sn_

mu_p = np.array([0.55, 0.07])
Sp_inv = np.diag([1/0.04, 1/0.003])

def Q_val(Ph, mn, Sn, ph):
    r = z - Ph @ mn
    T1 = -0.5/sigma2 * (r @ r + np.trace(Ph @ Sn @ Ph.T))
    dp = ph - mu_p
    return T1 - 0.5 * dp @ Sp_inv @ dp

print("\nQ vs p1:")
for p1v in [0.01,0.1,0.3,0.6,1.5,10]:
    Ph, mn_, Sn_ = build_Phi_post(p1v, p2_true)
    print(f"  p1={p1v:.1f}: Q={Q_val(Ph, mn_, Sn_, np.array([p1v,p2_true])):.3f}")

print("\nQ vs p2:")
for p2v in [0.002,0.06,0.08,0.4,0.6,1.5]:
    Ph, mn_, Sn_ = build_Phi_post(p1_true, p2v)
    print(f"  p2={p2v:.3f}: Q={Q_val(Ph, mn_, Sn_, np.array([p1_true,p2v])):.3f}")

# Estimated PDE solution from the reconstructed forcing field
u_hat = forward_solve(f_hat, p1_true, p2_true)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(
    f_true,
    aspect='auto',
    origin='lower',
    extent=[0, Lx, 0, T],
    cmap='viridis'
)
axes[0].set_title('True Forcing Field $f_{true}(x,t)$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    f_hat,
    aspect='auto',
    origin='lower',
    extent=[0, Lx, 0, T],
    cmap='viridis'
)
axes[1].set_title('Estimated Forcing Field $\hat{f}(x,t)$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    f_hat - f_true,
    aspect='auto',
    origin='lower',
    extent=[0, Lx, 0, T],
    cmap='RdBu_r'
)
axes[2].set_title('Forcing Error $\hat{f} - f_{true}$')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(
    u_true,
    aspect='auto',
    origin='lower',
    extent=[0, Lx, 0, T],
    cmap='plasma'
)
axes[0].scatter(x_grid[obs_xi], t_grid[obs_ti], c='white', s=15, edgecolor='black', label='observations')
axes[0].set_title('True PDE Solution $u_{true}(x,t)$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].legend(loc='upper right', fontsize='small')
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    u_hat,
    aspect='auto',
    origin='lower',
    extent=[0, Lx, 0, T],
    cmap='plasma'
)
axes[1].set_title('Estimated PDE Solution $\hat{u}(x,t)$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
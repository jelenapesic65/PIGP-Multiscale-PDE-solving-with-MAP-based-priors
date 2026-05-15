import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import minimize
from numpy.linalg import slogdet
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
import warnings
warnings.filterwarnings('ignore')

import numpy.random as npr


V_TRUE     = 5.0
KAPPA_TRUE = 0.01

N_OBS  = 100
N_C = 100
N_STAR = 100
N_BC = 30

NOISE_OBS = 1e-3
NOISE_BC  = 1e-3

def u1d(x, kappa, v):
    pi = np.pi
    denom = pi**2 * kappa**2 + v**2
    A  =  kappa / denom
    B  = -v     / (pi * denom)
    r  = v / kappa
    e  = np.exp(r)
    C2 = 2*B / (e - 1)
    C1 = -B - C2
    return C1 + C2*np.exp(r*x) + A*np.sin(pi*x) + B*np.cos(pi*x)

def u_exact(xy, kappa, v):
    #xy [N, 2]
    # u = u1d * sin(pi*y)
    x, y = xy[:, 0], xy[:, 1]
    u2d = u1d(x, kappa, v) * np.sin(np.pi * y)
    return u2d

def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f_rhs_2d(xy, kappa, v):
    """
    RHS such that  v*du/dx - kappa*(u_xx + u_yy) = f
    f(x,y) = sin(pi*x)*sin(pi*y)*(1 + kappa*pi^2)
             + kappa*pi^2 * u_1d(x)*sin(pi*y)   ... which simplifies to:
    f = sin(pi*x)*sin(pi*y) + kappa*pi^2*u_1d(x)*sin(pi*y)
       (the v*u'_1d - kappa*u''_1d part = sin(pi*x), the -kappa*(-pi^2)*u_1d*sin(pi*y) part)
    In practice we evaluate it directly.
    """
    x, y = xy[:, 0], xy[:, 1]
    u1   = u1d(x, kappa, v)
    # v*du/dx = v * u1d'(x) * sin(pi*y)
    # kappa*(u_xx + u_yy) = kappa*(u1d''(x)*sin(pi*y) - pi^2*u1d(x)*sin(pi*y))
    # => f = (v*u1d' - kappa*u1d'')*sin(pi*y) + kappa*pi^2*u1d(x)*sin(pi*y)
    #      = sin(pi*x)*sin(pi*y) + kappa*pi^2*u1d(x)*sin(pi*y)
    return (np.sin(np.pi * x) + kappa * np.pi**2 * u1) * np.sin(np.pi * y)

def _rbf_factors(p, q, lx, ly, sigma):
    """
    Return scalar factors needed to build all required derivative blocks.

    p : (M,2)  first set of points
    q : (N,2)  second set of points

    Returns arrays of shape (M,N):
        Kx, Ky, rx, ry  (and derived quantities)
    """
    rx  = p[:, 0:1] - q[None, :, 0]   # (M,N)  -- note: p[:,0:1] is (M,1)
    ry  = p[:, 1:2] - q[None, :, 1]

    # Fix indexing: p is (M,2), so p[:,0:1] shape=(M,1); q[:,0] shape=(N,)
    # Use explicit broadcasting
    rx  = p[:, 0][:, None] - q[:, 0][None, :]   # (M,N)
    ry  = p[:, 1][:, None] - q[:, 1][None, :]   # (M,N)

    lx2, ly2 = lx**2, ly**2

    Kx  = np.exp(-0.5 * rx**2 / lx2)
    Ky  = np.exp(-0.5 * ry**2 / ly2)
    K   = sigma**2 * Kx * Ky

    return K, Kx, Ky, rx, ry, lx2, ly2
def k2d(p, q, lx, ly, sigma):
    """Plain 2D product-RBF kernel, shape (M,N)."""
    K, *_ = _rbf_factors(p, q, lx, ly, sigma)
    return K

def k2d_L2(p, q, lx, ly, sigma, v, kappa):
    """
    L applied to second argument:  L_q k(p,q)
      = v * dk/dx2 - kappa*(d^2k/dx2^2 + d^2k/dy2^2)

    Note: dk/dx2 = -dk/dx1 (shift symmetry), so dk/dx2 = +rx/lx^2 * K
          d^2k/dx2^2 = d^2k/dx1^2 = (rx^2/lx^4 - 1/lx^2) * K
          d^2k/dy2^2 = (ry^2/ly^4 - 1/ly^2) * K
    """
    K, Kx, Ky, rx, ry, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)

    dK_dx2   =  (rx / lx2) * K
    d2K_dx22 = (rx**2 / lx2**2 - 1.0 / lx2) * K
    d2K_dy22 = (ry**2 / ly2**2 - 1.0 / ly2) * K

    return v * dK_dx2 - kappa * (d2K_dx22 + d2K_dy22)


def k2d_L1L2(p, q, lx, ly, sigma, v, kappa):
    
    K, Kx, Ky, rx, ry, lx2, ly2 = _rbf_factors(p, q, lx, ly, sigma)

    ex  = rx / lx2   # (M,N)
    ey  = ry / ly2

    
    axx  = ex**2 - 1.0/lx2    # d2K/dx1^2 / K  (same for dx2^2)
    ayy  = ey**2 - 1.0/ly2    # d2K/dy1^2 / K

    Lq = v * ex - kappa * (axx + ayy)

   

    dLq_dx1  = (v - 2*kappa*ex) / lx2
    d2Lq_dx1 = -2*kappa / lx2**2
    dLq_dy1  = -2*kappa*ey / ly2
    d2Lq_dy1 = -2*kappa / ly2**2

    dF_dx1   = (dLq_dx1 - ex*Lq) * K
    d2F_dx1  = (d2Lq_dx1 - 1.0/lx2 * Lq - 2*ex*dLq_dx1 + ex**2*Lq) * K
    d2F_dy1  = (d2Lq_dy1 - 1.0/ly2 * Lq - 2*ey*dLq_dy1 + ey**2*Lq) * K

    return v*dF_dx1 - kappa*(d2F_dx1 + d2F_dy1)


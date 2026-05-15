import numpy as np
import matplotlib.pyplot as plt

omega = np.linspace(-5, 5, 1000)

ell = 1.0

def matern_spectral(omega, nu, ell):
    return (2 * nu / ell**2 + (2 * np.pi * omega)**2) ** (-(nu + 0.5))

def se_spectral(omega, ell):
    return np.exp(-2 * (np.pi**2) * ell**2 * omega**2)

S_se = se_spectral(omega, ell)
S_m32 = matern_spectral(omega, nu=1.5, ell=ell)
S_m52 = matern_spectral(omega, nu=2.5, ell=ell)
S_m_inf_approx = matern_spectral(omega, nu=100, ell=ell)  # large nu ≈ SE

def normalize(S):
    return S / np.max(S)

S_se = normalize(S_se)
S_m32 = normalize(S_m32)
S_m52 = normalize(S_m52)
S_m_inf_approx = normalize(S_m_inf_approx)

plt.figure(figsize=(10, 6))

plt.plot(omega, S_se, label="SE (Gaussian decay)", linewidth=2)
plt.plot(omega, S_m32, label="Matérn ν=3/2 (polynomial tail)", linestyle='--')
plt.plot(omega, S_m52, label="Matérn ν=5/2 (steeper polynomial)", linestyle='--')
plt.plot(omega, S_m_inf_approx, label="Matérn ν→∞ (≈ SE)", linestyle=':')

plt.yscale('log')  # log scale to highlight tail behavior
plt.xlabel("Frequency ω")
plt.ylabel("Spectral Density S(ω) (normalized, log scale)")
plt.title("Spectral (Bochner) View of Kernels")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
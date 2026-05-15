import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

#the observed effect was for 20 points in [0, 0.9] and 10 points in [0.8, 1]

#something very interesting when I abruptly change the distribution of training points, in the midpoint an oscillation appears in the RBF solution, but not in the Matérn solution.
# This is because the RBF kernel is infinitely differentiable and thus very smooth, which can lead to overfitting and oscillations when the training data distribution changes abruptly.
#  The Matern kernel with ν=1.5 is less smooth and can better capture the underlying function without overfitting, resulting in a more stable solution even with abrupt changes in training data distribution.


def true_solution(x, Pe):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1) + np.sin(np.pi*x)

Pe = 500 

x_true = np.linspace(0, 1, 500).reshape(-1, 1)
y_true = true_solution(x_true, Pe).ravel()

np.random.seed(0)
x_train = np.linspace(0, 0.9, 50).reshape(-1, 1)
x_train = np.vstack([x_train, np.linspace(0.9, 1, 30).reshape(-1, 1)])  
print("Training points (x_train):", x_train.ravel())
y_train = true_solution(x_train, Pe).ravel()

kernel_rbf =  RBF(length_scale=0.5, length_scale_bounds=(1e-3, 1))
kernel_matern_32 = 1.0 * Matern(length_scale=0.2, nu=1.5, length_scale_bounds=(1e-3, 1))
kernel_matern_52 = 1.0 * Matern(length_scale=0.2, nu=2.5, length_scale_bounds=(1e-3, 1))

gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, alpha=1e-6,
                                   optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)
gp_matern_32 = GaussianProcessRegressor(kernel=kernel_matern_32, alpha=1e-6,
                                        optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)
gp_matern_52 = GaussianProcessRegressor(kernel=kernel_matern_52, alpha=1e-6,
                                        optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)

gp_rbf.fit(x_train, y_train)
gp_matern_32.fit(x_train, y_train)
gp_matern_52.fit(x_train, y_train)

print('Optimized kernels:')
print('RBF:', gp_rbf.kernel_)
print('Matérn 3/2:', gp_matern_32.kernel_)
print('Matérn 5/2:', gp_matern_52.kernel_)
print(gp_rbf.kernel_.get_params())

y_rbf, _ = gp_rbf.predict(x_true, return_std=True)
y_matern_32, _ = gp_matern_32.predict(x_true, return_std=True)
y_matern_52, _ = gp_matern_52.predict(x_true, return_std=True)


plt.figure(figsize=(8, 6))

plt.plot(x_true, y_true, 'k', label='True solution', linewidth=2)

plt.plot(x_true, y_rbf, 'r--', label='RBF')
plt.plot(x_true, y_matern_32, 'b--', label='Matérn 3/2 ')
plt.plot(x_true, y_matern_52, 'g--', label='Matérn 5/2')

plt.scatter(x_train, y_train, c='k', s=30, label='Training points')

plt.title(f'High-Pe Advection-Diffusion (Pe={Pe})')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
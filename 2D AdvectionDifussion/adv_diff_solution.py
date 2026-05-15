import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
Nx = 80
Ny = 80
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
dx = x[1] - x[0]

X, Y = np.meshgrid(x, y)

kappa_vals = np.linspace(0.05, 0.1, 1)
mu_vals = np.linspace(0.0, 2.0, 5)

solutions = {}

def solve_U(mu, kappa):
    A = np.zeros((Nx, Nx))
    b = np.sin(np.pi * x)

    for i in range(1, Nx-1):
        A[i, i-1] = -kappa/dx**2 - mu/(2*dx)
        A[i, i]   = 2*kappa/dx**2 + kappa*np.pi**2
        A[i, i+1] = -kappa/dx**2 + mu/(2*dx)

    A[0, 0] = 1
    A[-1, -1] = 1
    b[0] = 0
    b[-1] = 0

    U = np.linalg.solve(A, b)
    Z = np.outer(U, np.sin(np.pi * y)).T
    return Z

for i, kappa in enumerate(kappa_vals):
    for j, mu in enumerate(mu_vals):
        solutions[(i, j)] = solve_U(mu, kappa)

# Initial plot
init_kappa = 0
init_mu = 0

fig = go.Figure(
    data=[go.Surface(z=solutions[(init_kappa, init_mu)],
                     x=X, y=Y)]
)

# Create slider steps
steps = []
for i, kappa in enumerate(kappa_vals):
    for j, mu in enumerate(mu_vals):
        step = dict(
            method="update",
            args=[{"z": [solutions[(i, j)]]},
                  {"title": f"kappa={kappa:.2f}, mu={mu:.2f}"}],
            label=f"k={kappa:.2f}, μ={mu:.2f}"
        )
        steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Parameters: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    title="Advection-Diffusion Solution",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='u(x,y)'
    ),
    sliders=sliders
)

fig.show()
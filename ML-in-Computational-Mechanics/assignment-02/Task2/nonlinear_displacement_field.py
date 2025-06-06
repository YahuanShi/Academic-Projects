import torch
import matplotlib.pyplot as plt
import numpy as np


def helmholtz_free_energy(X):
    # Material parameters
    mu = 384.614
    lam = 576.923

    X.requires_grad_(True)

    # Define a nonlinear displacement field u(X)
    # u_x = X_x^2 + X_y^2
    # u_y = sin(X_x) + X_y^3
    u_x = X[:, 0] ** 2 + X[:, 1] ** 2
    u_y = torch.sin(X[:, 0]) + X[:, 1] ** 3
    u = torch.stack((u_x, u_y), dim=1)

    # Deformed coordinates x = X + u
    x = X + u

    # Compute gradient of x[:,0]
    grad_xx = torch.autograd.grad(x[:, 0].sum(), X, create_graph=True)[0][:, 0]
    grad_xy = torch.autograd.grad(x[:, 0].sum(), X, create_graph=True)[0][:, 1]

    # Compute gradient of x[:,1]
    grad_yx = torch.autograd.grad(x[:, 1].sum(), X, create_graph=True)[0][:, 0]
    grad_yy = torch.autograd.grad(x[:, 1].sum(), X, create_graph=True)[0][:, 1]

    N = X.shape[0]
    F = torch.zeros(N, 2, 2, dtype=X.dtype, device=X.device)
    F[:, 0, 0] = grad_xx
    F[:, 0, 1] = grad_xy
    F[:, 1, 0] = grad_yx
    F[:, 1, 1] = grad_yy

    J = torch.det(F)
    C = torch.matmul(F.transpose(1, 2), F)
    I1 = torch.diagonal(C, dim1=1, dim2=2).sum(dim=1)

    lnJ = torch.log(J)
    Psi = (lam / 2.) * (lnJ ** 2) - mu * lnJ + (mu / 2.) * (I1 - 2.)

    return Psi, x



# Generate test points
Xx = torch.linspace(0, 1, 50)
Xy = torch.linspace(0, 1, 50)
X_X, X_Y = torch.meshgrid(Xx, Xy, indexing='ij')
X_test = torch.stack((X_X.flatten(), X_Y.flatten()), dim=1)

# Compute free energy and deformed coordinates
Psi_values, x_deformed = helmholtz_free_energy(X_test)
Psi_values_np = Psi_values.detach().numpy()
x_deformed_np = x_deformed.detach().numpy()
X_ref_np = X_test.detach().numpy()

Psi_values_grid = Psi_values_np.reshape(50, 50)
X_ref_x_grid = X_ref_np[:, 0].reshape(50, 50)
X_ref_y_grid = X_ref_np[:, 1].reshape(50, 50)
X_def_x_grid = x_deformed_np[:, 0].reshape(50, 50)
X_def_y_grid = x_deformed_np[:, 1].reshape(50, 50)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Reference and deformed coordinate points
axs[0].plot(X_ref_x_grid.flatten(), X_ref_y_grid.flatten(), 'b.', alpha=0.5, label='Reference')
axs[0].plot(X_def_x_grid.flatten(), X_def_y_grid.flatten(), 'r.', alpha=0.5, label='Deformed')
axs[0].set_title('(a)Coordinates in reference and deformed state')
axs[0].set_xlabel('X_x')
axs[0].set_ylabel('X_y')
axs[0].legend()
axs[0].axis('equal')

# Right plot: Helmholtz free energy distribution
pcm = axs[1].pcolormesh(X_ref_x_grid, X_ref_y_grid, Psi_values_grid, cmap='plasma', shading='auto')
fig.colorbar(pcm, ax=axs[1], label='Helmholtz Free Energy (Pa)')
axs[1].set_title('(b)Helmholtz Free Energy Distribution')
axs[1].set_xlabel('X_x')
axs[1].set_ylabel('X_y')
axs[1].axis('equal')

plt.tight_layout()
plt.show()

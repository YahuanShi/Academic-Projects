import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def helmholtz_free_energy(X):
    """
    Given reference coordinates X, compute the Neo-Hookean free energy Ψ.
    Steps:
    1. Given the displacement field u(X) = [u_x, u_y], compute the deformed coordinates x = X + u.
    2. Use autograd to compute the deformation gradient F.
    3. From F, compute invariants I1 and J = det(F).
    4. Use the Neo-Hookean potential function to compute Ψ.
    """

    # Material parameters
    mu = 384.614
    lam = 576.923

    X.requires_grad_(True)

    # Define the nonlinear deformation field
    u_x = -0.5 * X[:, 0] + 0.25*(X[:, 1] ** 3)
    u_y = -0.25 * X[:, 0] - 0.5 * X[:, 1]
    u = torch.stack((u_x, u_y), dim=1)

    # Deformed coordinates
    x = X + u

    # Compute deformation gradient F components via autograd
    grad_x = torch.autograd.grad(x[:, 0].sum(), X, create_graph=False, retain_graph=True)[0]
    grad_y = torch.autograd.grad(x[:, 1].sum(), X, create_graph=False, retain_graph=False)[0]

    F = torch.zeros(X.shape[0], 2, 2, dtype=X.dtype, device=X.device)
    F[:, 0, 0] = grad_x[:, 0]
    F[:, 0, 1] = grad_x[:, 1]
    F[:, 1, 0] = grad_y[:, 0]
    F[:, 1, 1] = grad_y[:, 1]

    J = torch.det(F)
    C = torch.matmul(F.transpose(1, 2), F)
    I1 = torch.diagonal(C, dim1=1, dim2=2).sum(dim=1)  # I1 = tr(C)

    lnJ = torch.log(J)
    # Neo-Hookean free energy
    Psi = (lam / 2.) * (lnJ ** 2) - mu * lnJ + (mu / 2.) * (I1 - 2.)

    return Psi, x, F, I1, J

def generate_data(n_samples=512):
    """
    Randomly generate n_samples reference points X uniformly in [0,1]^2.
    For each point, compute Ψ, I1, J(I3), and return them as training data.
    """
    X_random = torch.rand(n_samples, 2)*1.0
    Psi, x_def, F, I1, J = helmholtz_free_energy(X_random)
    I3 = J**2  # Use J^2 as I3
    return I1.detach(), I3.detach(), Psi.detach(), X_random.detach(), x_def.detach()

class CANN(nn.Module):
    """
    CANN structure:
    Input: (I1, I3)
    Transform: (I1->I1-3), (I3->I3-1)
    Features: X1=(I1-3), X1², X2=(I3-1), X2²
              For each feature, apply identity and exp(x)-1, resulting in 8 features total
    Output: Linear layer to predict Ψ from these 8 features
    """
    def __init__(self):
        super(CANN, self).__init__()
        self.linear = nn.Linear(8, 1)

    def forward(self, I1, I3):
        X1 = I1 - 3.0
        X2 = I3 - 1.0

        X1_1 = X1
        X1_2 = X1**2
        X2_1 = X2
        X2_2 = X2**2

        def exp_minus_one(x):
            return torch.exp(x) - 1.0

        features = torch.stack([
            X1_1, exp_minus_one(X1_1),
            X1_2, exp_minus_one(X1_2),
            X2_1, exp_minus_one(X2_1),
            X2_2, exp_minus_one(X2_2)
        ], dim=1)

        Psi_pred = self.linear(features)
        return Psi_pred

# Main workflow
# 1. Generate training data
I1_train, I3_train, Psi_train, X_train, x_def_train = generate_data(n_samples=512)

# 2. Compute normalization factors
I1_max = torch.max(torch.abs(I1_train))
I3_max = torch.max(torch.abs(I3_train))
Psi_max = torch.max(torch.abs(Psi_train))

# Avoid division by zero
if I1_max == 0: I1_max = 1.0
if I3_max == 0: I3_max = 1.0
if Psi_max == 0: Psi_max = 1.0

# 3. Normalize the data
I1_train_norm = I1_train / I1_max
I3_train_norm = I3_train / I3_max
Psi_train_norm = Psi_train / Psi_max

I1_train_t = I1_train_norm.clone()
I3_train_t = I3_train_norm.clone()
Psi_train_t = Psi_train_norm.unsqueeze(1).clone()

# 4. Define model, optimizer, and loss function
model = CANN()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

# 5. Training
loss_history = []
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    Psi_pred_norm = model(I1_train_t, I3_train_t)
    # Use normalized Psi for training
    loss = criterion(Psi_pred_norm, Psi_train_t)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Normalized Loss: {loss.item()}")

# 6. Prepare test data
Xx = torch.linspace(0, 1, 50)
Xy = torch.linspace(0, 1, 50)
X_X, X_Y = torch.meshgrid(Xx, Xy, indexing='ij')
X_test = torch.stack((X_X.flatten(), X_Y.flatten()), dim=1)

Psi_true, x_def, F_test, I1_test, J_test = helmholtz_free_energy(X_test)
I3_test = J_test

# Normalize test data
I1_test_norm = I1_test / I1_max
I3_test_norm = I3_test / I3_max

# Model prediction (normalized -> denormalized)
with torch.no_grad():
    Psi_pred_norm_test = model(I1_test_norm, I3_test_norm).flatten()
Psi_pred_test = Psi_pred_norm_test * Psi_max  # Denormalization

# 7. Compute errors
Psi_true_np = Psi_true.detach().numpy()
Psi_pred_np = Psi_pred_test.detach().numpy()
error = Psi_pred_np - Psi_true_np
rel_error = error / (np.abs(Psi_true_np) + 1e-8)

Psi_values_true_grid = Psi_true_np.reshape(50, 50)
Psi_values_pred_grid = Psi_pred_np.reshape(50, 50)
Psi_error_grid = rel_error.reshape(50, 50)

X_ref_np = X_test.detach().numpy()
x_def_np = x_def.detach().numpy()
X_def_x_grid = x_def_np[:, 0].reshape(50, 50)
X_def_y_grid = x_def_np[:, 1].reshape(50, 50)

# 8. Visualization - loss curve
plt.figure(figsize=(6,4))
plt.plot(loss_history, label='Training Loss (Normalized)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve (Normalized Data)')
plt.legend()
plt.tight_layout()
plt.show()

# Reference and deformed configuration
fig, ax = plt.subplots(figsize=(6,5))
X_ref_x_grid = X_ref_np[:, 0].reshape(50, 50)
X_ref_y_grid = X_ref_np[:, 1].reshape(50, 50)
ax.plot(X_ref_x_grid.flatten(), X_ref_y_grid.flatten(), 'b.', alpha=0.5, label='Reference Config')
ax.plot(X_def_x_grid.flatten(), X_def_y_grid.flatten(), 'r.', alpha=0.5, label='Deformed Config')
ax.set_title('Reference and Deformed Configuration')
ax.set_xlabel('X_x or x_x')
ax.set_ylabel('X_y or x_y')
ax.legend()
ax.axis('equal')
plt.tight_layout()
plt.show()

# True, predicted, and error in the deformed configuration
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

pcm1 = axs[0].pcolormesh(X_def_x_grid, X_def_y_grid, Psi_values_true_grid, cmap='plasma', shading='auto')
axs[0].set_title('(a)True Helmholtz Free Energy (Deformed)')
axs[0].set_xlabel('x_x')
axs[0].set_ylabel('x_y')
fig.colorbar(pcm1, ax=axs[0])

pcm2 = axs[1].pcolormesh(X_def_x_grid, X_def_y_grid, Psi_values_pred_grid, cmap='plasma', shading='auto')
axs[1].set_title('(b)Predicted Helmholtz Free Energy (Deformed)')
axs[1].set_xlabel('x_x')
axs[1].set_ylabel('x_y')
fig.colorbar(pcm2, ax=axs[1])

pcm3 = axs[2].pcolormesh(X_def_x_grid, X_def_y_grid, Psi_error_grid, cmap='bwr', shading='auto')
axs[2].set_title('(c)Relative Error (Deformed)')
axs[2].set_xlabel('x_x')
axs[2].set_ylabel('x_y')
fig.colorbar(pcm3, ax=axs[2])

plt.tight_layout()
plt.show()

print("Average absolute error: ", np.mean(np.abs(error)))
print("Average relative error: ", np.mean(np.abs(rel_error)))

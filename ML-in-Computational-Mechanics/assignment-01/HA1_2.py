import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_num_threads(16)

plt.rcParams["text.usetex"] = False
plt.rcParams["lines.markersize"] = 3
plt.rcParams["font.size"] = 18

# Defines the size of a 2-dimensional area.
Lx = 2
Ly = 1
## Define the number of points in the grid (density)
Nx = 80 * Lx + 1#161
Ny = 80 * Ly + 1#81
shape1 = (Nx, Ny)

# Convert Nx and Ny to PyTorch tensors
Nx = torch.tensor(Nx)
Ny = torch.tensor(Ny)

#Calculate grid spacing in X and Y directions
dx = (Lx / (Nx - 1))#dx = Lx / (Nx - 1) = 2 / 160 = 0.0125
dy = (Ly / (Ny - 1))#dy = Ly / (Ny - 1) = 1 / 80 = 0.0125

# Generate the coordinate points of a two-dimensional grid
# Generate Nx evenly distributed points from 0 to Lx (i.e. 2) in the X direction
# Generate Ny evenly distributed points from 0 to Ly (i.e. 1) in the Y direction
X = torch.meshgrid(torch.linspace(0, Lx, Nx), torch.linspace(0, Ly, Ny), indexing='ij')
X = torch.cat((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), dim=1)

## Definition of deformation parameters
A = 0.2
B = 0.1
C = 0.05
D = 0.3
k_x = 3
k_y = 2

#Compute the deformation field u
u = torch.stack((A * torch.sin(k_x * X[:, 0]) + B * X[:, 1]**2,
                  C * X[:, 0]**2 + D * torch.cos(k_y * X[:, 1])), dim=1)

#Calculate the coordinates after deformation
X_deformed = X + u

# Calculate the displacement of each point
displacement_magnitude = torch.sqrt(torch.sum(u ** 2, dim=1))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

#Scatter plot of original grid
ax1.scatter(X[:, 0].numpy(), X[:, 1].numpy(), color='black', s=10, marker='o')
ax1.set_title("Original Field X")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")

# Plot a scatter plot of the deformed mesh and colour it according to the magnitude of the displacement.
scatter = ax2.scatter(X_deformed[:, 0].numpy(), X_deformed[:, 1].numpy(),
                      c=displacement_magnitude.numpy(), cmap='viridis', s=10, marker='o')
ax2.set_title("Deformed Field X + u")
ax2.set_xlabel("X-axis")
ax2.set_ylabel("Y-axis")

# Add a colour bar to indicate the magnitude of the displacement.
plt.colorbar(scatter, ax=ax2, label="Displacement Magnitude")

# Adjust the layout
plt.tight_layout()
plt.show()
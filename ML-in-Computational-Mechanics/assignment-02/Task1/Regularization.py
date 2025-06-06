# Import necessary libraries
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)
device = torch.device("cpu")

model_file = "mymodel_reduced.torch"

# Generate training data
samples = 1000
sample_min = -5
sample_max = 5
sample_span = sample_max - sample_min

batch_size = 8
hidden_dim = 128  # Reduced hidden layer neurons
input_dim = 2
output_dim = 1
epochs = 2000
lr = 0.001

criterion = nn.MSELoss(reduction="mean")

# Generate training data with noise
train_x = (sample_span * torch.rand(samples, 2) + sample_min * torch.ones(samples, 2))
train_y = torch.sum(train_x ** 2, dim=1, keepdim=True)
noise_level = 10
noise = torch.randn_like(train_y) * noise_level
train_y_noisy = train_y + noise

# Generate testing data
num_test_points = 50
x_vals = torch.linspace(sample_min - 0.5 * sample_span, sample_max + 0.5 * sample_span, num_test_points)
y_vals = torch.linspace(sample_min - 0.5 * sample_span, sample_max + 0.5 * sample_span, num_test_points)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
X_flat = X.flatten()
Y_flat = Y.flatten()
test_x = torch.stack((X_flat, Y_flat), dim=1)
test_y = torch.sum(test_x ** 2, dim=1, keepdim=True)

# Create data loader
train_data = TensorDataset(train_x, train_y_noisy)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)

# Define neural network with reduced layers
class MLNetReduced(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(MLNetReduced, self).__init__()
        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Single hidden layer
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Add Dropout
            nn.Linear(hidden_dim, output_dim)  # Directly map to output
        )

    def forward(self, x):
        return self.fcnn1(x)

# Evaluate function
def evaluate(model, test_x, test_y):
    with torch.no_grad():
        model.eval()
        out = model(test_x.to(device))
        loss = criterion(out, test_y.to(device)).item()
        return out.cpu().detach().numpy(), test_y.cpu().detach().numpy(), loss

# Plotting function
def eval_and_plot(model):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    net_outputs_test, targets_test, test_loss = evaluate(model, test_x, test_y)

    X_plot = test_x[:, 0].cpu().numpy()
    Y_plot = test_x[:, 1].cpu().numpy()
    Z_true = targets_test.flatten()
    Z_pred = net_outputs_test.flatten()

    num_test_points = int(np.sqrt(len(X_plot)))
    X_plot = X_plot.reshape((num_test_points, num_test_points))
    Y_plot = Y_plot.reshape((num_test_points, num_test_points))
    Z_true = Z_true.reshape((num_test_points, num_test_points))
    Z_pred = Z_pred.reshape((num_test_points, num_test_points))

    ax.plot_surface(X_plot, Y_plot, Z_true, cmap='viridis', alpha=0.5)
    ax.plot_surface(X_plot, Y_plot, Z_pred, cmap='plasma', alpha=0.5)

    ax.scatter(train_x[:, 0].cpu().numpy(), train_x[:, 1].cpu().numpy(), train_y_noisy.cpu().numpy().flatten(),
               color='r', label='Training Data', marker='^')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Training function
def train_with_reduced_layers(train_loader, learn_rate, epochs, weight_decay=1e-5, dropout_rate=0.2):
    model = MLNetReduced(input_dim, hidden_dim, output_dim, dropout_rate)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)  # L2 Regularization

    avg_losses = torch.zeros(epochs)
    avg_test_losses = torch.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        avg_loss = 0.0

        for x, label in train_loader:
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = criterion(out, label.to(device))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_losses[epoch] = avg_loss / len(train_loader)
        with torch.no_grad():
            model.eval()
            out_test = model(test_x.to(device))
            test_loss = criterion(out_test, test_y.to(device)).item()
            avg_test_losses[epoch] = test_loss

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_losses[epoch]:.4f}, Test Loss: {test_loss:.4f}")
            eval_and_plot(model)

    plt.figure(figsize=(12, 8))
    plt.plot(avg_losses, label='Training Loss')
    plt.plot(avg_test_losses, label='Test Loss')
    plt.title("Training and Testing Loss with Regularization")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    torch.save(model, model_file)
    return model

# Train and evaluate
model = train_with_reduced_layers(train_loader, lr, epochs)
model = torch.load(model_file)
eval_and_plot(model)

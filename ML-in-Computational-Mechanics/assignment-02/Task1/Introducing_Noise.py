# Import matplotlib for plotting purposes
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt



import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)

device = torch.device("cpu")


model_file = "mymodel.torch"

# Generate training data
samples = 1000
sample_min = -5
sample_max = 5
sample_span = sample_max - sample_min

batch_size = 8  # Number of samples before optimizer update
hidden_dim = 512  # Increase number of neurons in the hidden layer to increase model capacity
input_dim = 2
output_dim = 1

epochs = 5000
lr = 0.001

criterion = nn.MSELoss(reduction="mean")

# Generate training data and add noise
train_x = (sample_span * torch.rand(samples, 2) + sample_min * torch.ones(samples, 2))
train_y = torch.sum(train_x ** 2, dim=1, keepdim=True)

# Add noise
noise_level = 10  # Standard deviation of noise
noise = torch.randn_like(train_y) * noise_level
train_y_noisy = train_y + noise

# Print training data
print("train_x", train_x)
print("train_y_noisy", train_y_noisy)

# Generate test data
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

# Define neural network model
class MLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLNet, self).__init__()
        self.hidden_dim = hidden_dim

        # Increase number of hidden layers for a deeper network
        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out = self.fcnn1(x)
        return out


def evaluate(model, test_x, test_y):
    with torch.no_grad():
        model.eval()
        outputs = []
        targets = []
        testlosses = []

        out = model(test_x.to(device))
        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.cpu().detach().numpy())
        testlosses.append(criterion(out, test_y.to(device)).item())

    return outputs, targets, testlosses

# plotting function
def eval_and_plot(model):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use the model to predict on test data
    net_outputs_test, targets_test, testlosses = evaluate(model, test_x, test_y)
    X_plot = test_x[:, 0].cpu().numpy()
    Y_plot = test_x[:, 1].cpu().numpy()
    Z_true = targets_test[0].flatten()
    Z_pred = net_outputs_test[0].flatten()

    # Reshape for surface plot
    num_test_points = int(np.sqrt(len(X_plot)))
    X_plot = X_plot.reshape((num_test_points, num_test_points))
    Y_plot = Y_plot.reshape((num_test_points, num_test_points))
    Z_true = Z_true.reshape((num_test_points, num_test_points))
    Z_pred = Z_pred.reshape((num_test_points, num_test_points))

    # Plot surface of the true function
    ax.plot_surface(X_plot, Y_plot, Z_true, cmap='viridis', alpha=0.5)

    # Plot surface of the model predictions
    ax.plot_surface(X_plot, Y_plot, Z_pred, cmap='plasma', alpha=0.5)

    # Plot training data
    X_train = train_x[:, 0].cpu().numpy()
    Y_train = train_x[:, 1].cpu().numpy()
    Z_train = train_y_noisy.cpu().numpy().flatten()
    ax.scatter(X_train, Y_train, Z_train, color='r', label='Training Data', marker='^')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

# Define training function
def train(train_loader, learn_rate, EPOCHS):
    # Instantiate the NN
    model = MLNet(input_dim, hidden_dim, output_dim)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    print("Starting Training of the model")

    # Save training and test losses for each epoch
    avg_losses = torch.zeros(EPOCHS)
    avg_test_losses = torch.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        avg_loss = 0.0
        counter = 0

        for x, label in train_loader:
            counter += 1
            model.zero_grad()

            out = model(x.to(device))
            loss = criterion(out, label.to(device))

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if counter % 100 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {:.4f}".format(
                    epoch+1, counter, len(train_loader), avg_loss / counter))

        # After each epoch, evaluate the model on the test set
        with torch.no_grad():
            model.eval()
            out_test = model(test_x.to(device))
            test_loss = criterion(out_test, test_y.to(device)).item()
            avg_test_losses[epoch] = test_loss

        avg_losses[epoch] = avg_loss / len(train_loader)

        print("Epoch {}/{} Done, Total Loss: {:.4f}, Test Loss: {:.4f}".format(
            epoch+1, EPOCHS, avg_losses[epoch], test_loss))

        # Every 1000 epochs, evaluate and plot the model
        if (epoch+1) % 1000 == 0:
            eval_and_plot(model)

    # Plot training and test loss curves
    plt.figure(figsize=(12, 8))
    plt.plot(avg_losses, label='Training Loss')
    plt.plot(avg_test_losses, label='Test Loss')
    plt.title("Train and Test Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the model
    torch.save(model, model_file)

    return model

# Start training the model
model = train(train_loader, lr, epochs)

# Load saved model and evaluate
model = torch.load(model_file)
eval_and_plot(model)

print(model)

# Print model weights and biases
print("Layer 1 weights:", model.fcnn1[0].weight)
print("Layer 1 bias:", model.fcnn1[0].bias)


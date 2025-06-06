
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt



import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


torch.set_default_dtype(torch.float32)  # 64)
torch.set_num_threads(4)
device = torch.device("cpu")
print(f"Using {device} device")
device = torch.device(device)


model_file = "mymodel.torch"


## The variables below are used to control this sampling
samples = 1000
sample_min = -5
sample_max = 5
sample_span = sample_max - sample_min


batch_size = 8
hidden_dim = 64
input_dim = 2
output_dim = 1

epochs = 1600
lr = 0.001

criterion = nn.MSELoss(reduction="mean")

## create the training data.
## We sample from the interval and pass it to a function, here z = x^2 + y^2
train_x = (sample_span * torch.rand(samples, 2) + sample_min * torch.ones(samples, 2))
train_y = torch.sum(train_x ** 2, dim=1, keepdim=True)

print("train_x", train_x)
print("train_y", train_y)

## create the test data.
## We choose points outside the training interval to test extrapolation and we choose them more densely.
num_test_points = 50
x_vals = torch.linspace(sample_min - 0.5 * sample_span, sample_max + 0.5 * sample_span, num_test_points)
y_vals = torch.linspace(sample_min - 0.5 * sample_span, sample_max + 0.5 * sample_span, num_test_points)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
X_flat = X.flatten()
Y_flat = Y.flatten()
test_x = torch.stack((X_flat, Y_flat), dim=1)
test_y = torch.sum(test_x ** 2, dim=1, keepdim=True)

train_data = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)

## Define the neural network model
class MLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    ###  forward pass
    def forward(self, x):
        out = self.fcnn1(x)
        return out

## Define evaluation function
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

## Define a function to evaluate and plot the model
def eval_and_plot(model):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ## Call the network on the test data
    net_outputs_test, targets_test, testlosses = evaluate(model, test_x, test_y)
    X_plot = test_x[:, 0].cpu().numpy()
    Y_plot = test_x[:, 1].cpu().numpy()
    Z_true = targets_test[0].flatten()
    Z_pred = net_outputs_test[0].flatten()

    ## Reshape for surface plot
    num_test_points = int(np.sqrt(len(X_plot)))
    X_plot = X_plot.reshape((num_test_points, num_test_points))
    Y_plot = Y_plot.reshape((num_test_points, num_test_points))
    Z_true = Z_true.reshape((num_test_points, num_test_points))
    Z_pred = Z_pred.reshape((num_test_points, num_test_points))

    ## Plot the true function surface
    ax.plot_surface(X_plot, Y_plot, Z_true, cmap='viridis', alpha=0.5, label='Target')

    ## Plot the predicted function surface
    ax.plot_surface(X_plot, Y_plot, Z_pred, cmap='plasma', alpha=0.5, label='Learned')

    ## Also plot the training data
    net_outputs_train, targets_train, _ = evaluate(model, train_x, train_y)
    X_train = train_x[:, 0].cpu().numpy()
    Y_train = train_x[:, 1].cpu().numpy()
    Z_train = targets_train[0].flatten()
    ax.scatter(X_train, Y_train, Z_train, color='r', label='Training Data', marker='^')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

## Define a training function
def train(train_loader, learn_rate, EPOCHS):

    # Instantiate the NN
    model = MLNet(input_dim, hidden_dim, output_dim)
    model.to(device)

    ## Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    print("Starting Training of the model")

    avg_losses = torch.zeros(EPOCHS)

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

            if counter % 20 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {} = {} + {}".format(epoch, counter,
                                                                                                    len(train_loader),
                                                                                                    avg_loss / counter,
                                                                                                    loss.item(), 0))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        avg_losses[epoch] = avg_loss / len(train_loader)

        ## Every 200 epochs, evaluate and plot the model
        if epoch % 200 == 0:
            eval_and_plot(model)

    ## Plot the training loss curve
    plt.figure(figsize=(12, 8))
    plt.plot(avg_losses, "x-")
    plt.title("Train loss (MSE, reduction=mean, averaged over epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.grid(visible=True, which='both', axis='both')
    plt.show()

    ## Save the trained model
    torch.save(model, model_file)

    return model

## Start training
model = train(train_loader, lr, epochs)

## Test loading the model and evaluate
model = torch.load(model_file)
eval_and_plot(model)

print(model)

# Print the weights and biases of the model
print("Layer 1 weights:", model.fcnn1[0].weight)
print("Layer 1 bias:", model.fcnn1[0].bias)
print("Layer 2 weights:", model.fcnn1[2].weight)
print("Layer 2 bias:", model.fcnn1[2].bias)

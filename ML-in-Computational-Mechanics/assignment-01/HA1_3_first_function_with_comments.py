import torch  # Import PyTorch library for building neural networks
import torch.nn as nn  # Import the neural network module from PyTorch
from torch.autograd import Variable  # Import Variable from PyTorch to wrap tensors

# Define a simple neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # Initialize the parent class nn.Module
        self.layer = torch.nn.Linear(1, 1)  # Define a linear layer (1 input, 1 output)

    def forward(self, x):
        x = self.layer(x)  # Forward propagation: pass input through the linear layer
        return x

# Create an instance of the Net class
net = Net()
print(net)  # Print the structure of the neural network


import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations


x = np.random.rand(100)  # Generate 100 random values between 0 and 1 as input data x
# Generate corresponding output data using a nonlinear function
y = np.sin(x) * np.power(x, 3) + 3 * x + np.random.rand(100) * 0.8  # Includes nonlinear function and noise
# np.sin(x) calculates the sine value of each element in x.
# np.power(x, 3) raises each element of x to the power of 3, representing x^3, enhancing the nonlinearity.
# Multiply np.sin(x) and np.power(x, 3) to obtain a new array. This part represents a complex nonlinear relationship, combining sine variation with a cubic term.
# np.random.rand(100) * 0.8 multiplies these random values by 0.8 to add some noise, simulating real-world data uncertainty.
# This code generates data y = sin(x) * x^3 + 3x + a, where a is the noise.

# Visualize the generated data
plt.scatter(x, y)
plt.show()

# Convert numpy arrays to PyTorch tensors and reshape them for the model
x = torch.from_numpy(x.reshape(-1, 1)).float()  # Convert numpy array x to a tensor and reshape it into two dimensions
y = torch.from_numpy(y.reshape(-1, 1)).float()  # Convert numpy array y to a tensor and reshape it into two dimensions
print(x, y)  # Print the tensors
# Use torch.from_numpy() to convert NumPy arrays into PyTorch tensors.
# x.reshape(-1, 1) and y.reshape(-1, 1) reshape the one-dimensional arrays into column vectors suitable for model input
# for example, x = np.array([0.1, 0.2, 0.3, 0.4]) becomes:
# x = [[0.1],
#      [0.2],
#      [0.3],
#      [0.4]]
# .float() converts the data type to float to make it compatible with PyTorch models.

# Define the optimizer and loss function for training
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()
# torch.optim.SGD: Define a stochastic gradient descent optimizer to update the model parameters. lr=0.2: learning rate is 0.2.
# torch.nn.MSELoss(): Define the loss function as Mean Squared Error (MSE), used to evaluate the error between predictions and true values.

# Wrap the tensors in Variables for compatibility with PyTorch
inputs = Variable(x)
outputs = Variable(y)

# Training loop
for i in range(250):  # Train for 250 epochs
    prediction = net(inputs)  # Get predictions from the network
    loss = loss_func(prediction, outputs)  # Calculate the loss between predictions and true values
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Backpropagate to compute gradients
    optimizer.step()  # Update the network's weights using the optimizer

    # Visualize the training process every 10 iterations
    if i % 10 == 0:
        plt.cla()  # Clear the previous plot
        plt.scatter(x.data.numpy(), y.data.numpy())  # Plot the real data points
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)  # Plot the model's predictions
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})  # Display the current loss value
        plt.pause(0.1)  # Pause for 0.1 seconds to display the dynamic training process

plt.show()  # Show the final plot after training

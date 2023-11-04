import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from network import NN_2_parameters
from data import data_generator


# Generate data (sphere model)
f_1_d_2_generator = data_generator(suite_name="bbob", function=1, dimension=2, instance=1)
data_x, data_y = f_1_d_2_generator.generate(data_size=200)

# Define the model function
def model(x, y):
    network = NN_2_parameters()
    new_weight = torch.FloatTensor([[x, y]])
    new_bias = torch.FloatTensor([0.5])
    network.fc1.weight = nn.Parameter(new_weight)
    network.fc1.bias = nn.Parameter(new_bias)
    return network

# Define the objective function
criterion = nn.MSELoss()
def loss(x, y):
    network = model(x, y)
    predicted_y = network(data_x)
    return criterion(data_y, predicted_y).detach().numpy()

# Define the range of parameter values
x_range = np.linspace(-100, 100, 2000)
y_range = np.linspace(-100, 100, 2000)

# Create a meshgrid of parameter values
X, Y = np.meshgrid(x_range, y_range)

# Calculate the loss for each combination of parameters
Z = np.vectorize(loss)(X, Y)

# Create a heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Loss Landscape of Neural Network')
plt.show()
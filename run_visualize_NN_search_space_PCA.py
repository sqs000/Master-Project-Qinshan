import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Loss versus NN Parameters
def train_neural_network(network, parameters, evaluation_x, evaluation_y):
    # assign the network with specific parameters
    new_params = torch.split(torch.tensor(parameters), [p.numel() for p in network.parameters()])
    with torch.no_grad():
        for param, new_param_value in zip(network.parameters(), new_params):
            param.data.copy_(new_param_value.reshape(param.data.shape))
    # evaluate the network with the parameters
    predicted_y = network(evaluation_x)
    criterion = nn.MSELoss()
    # return the loss
    return criterion(evaluation_y, predicted_y).item()


# Generate NN parameters randomly
def generate_random_arrays(length, num_arrays, value_range):
    arrays = []
    for _ in range(num_arrays):
        array = np.random.uniform(low=value_range[0], high=value_range[1], size=length)
        arrays.append(array)
    return np.array(arrays)


# Generate data
f_3_d_3_generator = data_generator(suite_name="bbob", function=3, dimension=3, instance=1)
data_x, data_y = f_3_d_3_generator.generate(data_size=300)
# Generate NN parameters and corresponding loss
num_samples = 10000
model_NN = hidden2_FNN(3, 50, 20, 1)
num_parameters = sum(p.numel() for p in model_NN.parameters())
param_values = generate_random_arrays(num_parameters, num_samples, [-100, 100])
loss_values = np.array([train_neural_network(model_NN, sample_values, data_x, data_y) for sample_values in param_values])

# Use PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
param_values_pca = pca.fit_transform(param_values)

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(param_values_pca[:, 0], param_values_pca[:, 1], c=loss_values, cmap='viridis', s=50, alpha=0.7)

# Customize the plot
plt.title('PCA Visualization of Neural Network Parameter Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Add colorbar
cbar = plt.colorbar(scatter, label='Loss')

# Show the plot
plt.show()


# # Perform PCA for dimensionality reduction to 3D
# pca = PCA(n_components=3)
# param_values_pca = pca.fit_transform(param_values)

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(param_values_pca[:, 0], param_values_pca[:, 1], param_values_pca[:, 2], c=loss_values, cmap='viridis', s=30)

# # Customize the plot
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# ax.set_title('Neural Network Parameter Space')

# # Add colorbar
# cbar = fig.colorbar(sc, ax=ax, label='Loss')

# # Show the plot
# plt.show()
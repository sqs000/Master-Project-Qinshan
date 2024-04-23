import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def init_weights(weight=0.01, bias=0.01):
    def function(model): 
        if isinstance(model, nn.Linear):
            model.weight.data.fill_(weight)
            model.bias.data.fill_(bias)
    return function


def euclidean_distance(model1, model2):
    params1 = torch.cat([p.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.view(-1) for p in model2.parameters()])
    return torch.norm(params1 - params2).item()

def vector_euclidean_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def heatmap_2d(x_range, y_range, function, x_label="Parameter 1", y_label="Parameter 2", title="Loss Landscape of Neural Network", colorbar_label="Loss", cmap='viridis', figsize=(8, 6)):

    # Define the range of parameter values
    x_range = np.linspace(x_range[0], x_range[1], 400)
    y_range = np.linspace(y_range[0], y_range[1], 400)

    # Create a meshgrid of parameter values
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the loss for each combination of parameters
    Z = np.vectorize(function)(X, Y)

    plt.figure(figsize=figsize)
    plt.pcolormesh(X, Y, Z, cmap=cmap)
    plt.colorbar(label=colorbar_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def assign_param(network, parameters):
    """ Assign a network with specific parameters."""
    new_params = torch.split(torch.tensor(parameters), [p.numel() for p in network.parameters()])
    with torch.no_grad():
        for param, new_param_value in zip(network.parameters(), new_params):
            param.data.copy_(new_param_value.reshape(param.data.shape))


def flatten_list(lst):
        result = []
        for el in lst:
            if isinstance(el, list):
                result.extend(flatten_list(el))
            else:
                result.append(el)
        return result

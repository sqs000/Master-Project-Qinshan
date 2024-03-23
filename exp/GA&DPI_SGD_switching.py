import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import numpy as np
import torch
import torch.nn as nn
from algorithm import SGDOpt
from network import hidden2_FNN
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load data
    data_x = torch.load("results/GA/data_x.pt")
    data_y = torch.load("results/GA/data_y.pt")

    # load GA&DPI params
    ga_dynamic_params = np.load("results/GA/ga_dynamic_params.npy")
    ga_dynamic_results = np.load("results/GA/ga_dynamic_results.npy")
    ga_dynamic_params_best = ga_dynamic_params[np.argmin(ga_dynamic_results)]

    # function to assign a network with specific parameters
    def assign_param(network, parameters):
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))

    # build GA&DPI network
    ga_dynamic_best_network = hidden2_FNN(2, 50, 20, 1)
    ga_dynamic_best_network.to(torch.device("cuda"))
    assign_param(ga_dynamic_best_network, ga_dynamic_params_best)

    # SGD training start from the GA&DPI network
    best_params, best_loss, epochs, epoch_losses = SGDOpt(network=ga_dynamic_best_network, data_x=data_x, data_y=data_y, criterion=nn.MSELoss(), n_epochs=1000, sgd_lr=0.00001)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses)
    plt.title('The NN training loss curve using SGD starting from the result of GA_dynamic')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.show()


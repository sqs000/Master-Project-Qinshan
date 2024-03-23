import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import numpy as np
import torch
from data import data_generator
from network import hidden2_FNN
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # data generator for the true function
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)

    # build the input mesh grid for plotting
    x1_values = np.linspace(-5, 5, 100)
    x2_values = np.linspace(-5, 5, 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)
    input_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
    # function to plot 3D surface
    def plot_3d_surface(ax, title, x, y, z):
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8, edgecolor='k')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_title(title)
    # function to assign a network with specific parameters
    def assign_param(network, parameters):
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))

    # read results from experiments
    # read network parameters
    sgd_params = np.load("results/GA/sgd_params.npy")
    ga_params = np.load("results/GA/ga_params.npy")
    ga_sharing_params = np.load("results/GA/ga_sharing_params.npy")
    ga_dynamic_params = np.load("results/GA/ga_dynamic_params.npy")
    # read losses corresponding to NN parameters
    sgd_results = np.load("results/GA/sgd_results.npy")
    ga_results = np.load("results/GA/ga_results.npy")
    ga_sharing_results = np.load("results/GA/ga_sharing_results.npy")
    ga_dynamic_results = np.load("results/GA/ga_dynamic_results.npy")
    # obtain the best parameters according to their losses
    sgd_params_best = sgd_params[np.argmin(sgd_results)]
    ga_params_best = ga_params[np.argmin(ga_results)]
    ga_sharing_params_best = ga_sharing_params[np.argmin(ga_sharing_results)]
    ga_dynamic_params_best = ga_dynamic_params[np.argmin(ga_dynamic_results)]

    # network building
    # sgd
    sgd_best_network = hidden2_FNN(2, 50, 20, 1)
    assign_param(sgd_best_network, sgd_params_best)
    # ga
    ga_best_network = hidden2_FNN(2, 50, 20, 1)
    assign_param(ga_best_network, ga_params_best)
    # ga_sharing
    ga_sharing_best_network = hidden2_FNN(2, 50, 20, 1)
    assign_param(ga_sharing_best_network, ga_sharing_params_best)
    # ga_dynamic
    ga_dynamic_best_network = hidden2_FNN(2, 50, 20, 1)
    assign_param(ga_dynamic_best_network, ga_dynamic_params_best)

    # start plotting
    fig = plt.figure(figsize=(20, 5))
    # fig1
    ax0_sgd = fig.add_subplot(151, projection='3d')
    with torch.no_grad():
        sgd_output = sgd_best_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax0_sgd, 'SGD best result', x1_mesh, x2_mesh, sgd_output)
    # fig2
    ax1_ga = fig.add_subplot(152, projection='3d')
    with torch.no_grad():
        ga_output = ga_best_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    ga_output = ga_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax1_ga, 'GA best result', x1_mesh, x2_mesh, ga_output)
    # fig3
    ax2_ga_sharing = fig.add_subplot(153, projection='3d')
    with torch.no_grad():
        ga_sharing_output = ga_sharing_best_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    ga_sharing_output = ga_sharing_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax2_ga_sharing, 'GA_sharing best result', x1_mesh, x2_mesh, ga_sharing_output)
    # fig4
    ax3_ga_dynamic = fig.add_subplot(154, projection='3d')
    with torch.no_grad():
        ga_dynamic_output = ga_dynamic_best_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    ga_dynamic_output = ga_dynamic_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax3_ga_dynamic, 'GA_dynamic best result', x1_mesh, x2_mesh, ga_dynamic_output)
    # fig5
    ax4_true = fig.add_subplot(155, projection='3d')
    true_function_values = np.array([f_3_d_2_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax4_true, 'True Function', x1_mesh, x2_mesh, true_function_values)
    # show
    plt.show()

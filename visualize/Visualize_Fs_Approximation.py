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
from utils import assign_param


if __name__ == "__main__":
    # data generator for the true function
    f1_generator = data_generator(suite_name="bbob", function=1, dimension=2, instance=1)
    f3_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    f7_generator = data_generator(suite_name="bbob", function=7, dimension=2, instance=1)
    f13_generator = data_generator(suite_name="bbob", function=13, dimension=2, instance=1)
    f16_generator = data_generator(suite_name="bbob", function=16, dimension=2, instance=1)
    f22_generator = data_generator(suite_name="bbob", function=22, dimension=2, instance=1)


    # build the input mesh grid for plotting
    x1_values = np.linspace(-5, 5, 100)
    x2_values = np.linspace(-5, 5, 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)
    input_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
    
    # function to plot 3D surface
    def plot_3d_surface(ax, title, x, y, z):
        ax.plot_surface(x, y, z, cmap='winter', alpha=0.8, linewidth=0)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_title(title)
        ax.grid(False)

    # read network parameters
    f1_sgd_params = np.load("results\SGD_Fs_for_visualization\BBOB-1_SGD_f-ind.npy")
    f3_sgd_params = np.load("results\SGD_Fs_for_visualization\BBOB-3_SGD_f-ind.npy")
    f7_sgd_params = np.load("results\SGD_Fs_for_visualization\BBOB-7_SGD_f-ind.npy")
    f13_sgd_params = np.load("results\SGD_Fs_for_visualization\BBOB-13_SGD_f-ind.npy")
    f16_sgd_params = np.load("results\SGD_Fs_for_visualization\BBOB-16_SGD_f-ind.npy")
    f22_sgd_params = np.load("results\SGD_Fs_for_visualization\BBOB-22_SGD_f-ind.npy")
    
    # network building
    f1_sgd_network = hidden2_FNN(2, 50, 20, 1)
    f3_sgd_network = hidden2_FNN(2, 50, 20, 1)
    f7_sgd_network = hidden2_FNN(2, 50, 20, 1)
    f13_sgd_network = hidden2_FNN(2, 50, 20, 1)
    f16_sgd_network = hidden2_FNN(2, 50, 20, 1)
    f22_sgd_network = hidden2_FNN(2, 50, 20, 1)
    assign_param(f1_sgd_network, f1_sgd_params)
    assign_param(f3_sgd_network, f3_sgd_params)
    assign_param(f7_sgd_network, f7_sgd_params)
    assign_param(f13_sgd_network, f13_sgd_params)
    assign_param(f16_sgd_network, f16_sgd_params)
    assign_param(f22_sgd_network, f22_sgd_params)
    
    # start plotting
    fig = plt.figure(figsize=(24, 10))
    # f1 - sgd
    ax1_sgd = fig.add_subplot(2, 6, 1, projection='3d')
    with torch.no_grad():
        sgd_output = f1_sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax1_sgd, 'F1 SGD result', x1_mesh, x2_mesh, sgd_output)
    # f3 - sgd
    ax3_sgd = fig.add_subplot(2, 6, 2, projection='3d')
    with torch.no_grad():
        sgd_output = f3_sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax3_sgd, 'F3 SGD result', x1_mesh, x2_mesh, sgd_output)
    # f7 - sgd
    ax7_sgd = fig.add_subplot(2, 6, 3, projection='3d')
    with torch.no_grad():
        sgd_output = f7_sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax7_sgd, 'F7 SGD result', x1_mesh, x2_mesh, sgd_output)
    # f13 - sgd
    ax13_sgd = fig.add_subplot(2, 6, 4, projection='3d')
    with torch.no_grad():
        sgd_output = f13_sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax13_sgd, 'F13 SGD result', x1_mesh, x2_mesh, sgd_output)
    # f16 - sgd
    ax16_sgd = fig.add_subplot(2, 6, 5, projection='3d')
    with torch.no_grad():
        sgd_output = f16_sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax16_sgd, 'F16 SGD result', x1_mesh, x2_mesh, sgd_output)
    # f22 - sgd
    ax22_sgd = fig.add_subplot(2, 6, 6, projection='3d')
    with torch.no_grad():
        sgd_output = f22_sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_output = sgd_output.reshape(x1_mesh.shape)
    plot_3d_surface(ax22_sgd, 'F22 SGD result', x1_mesh, x2_mesh, sgd_output)
    
    # f1 - true
    ax1_true = fig.add_subplot(2, 6, 7, projection='3d')
    true_function_values = np.array([f1_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax1_true, 'F1 True Function', x1_mesh, x2_mesh, true_function_values)
    # f3 - true
    ax3_true = fig.add_subplot(2, 6, 8, projection='3d')
    true_function_values = np.array([f3_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax3_true, 'F3 True Function', x1_mesh, x2_mesh, true_function_values)
    # f7 - true
    ax7_true = fig.add_subplot(2, 6, 9, projection='3d')
    true_function_values = np.array([f7_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax7_true, 'F7 True Function', x1_mesh, x2_mesh, true_function_values)
    # f13 - true
    ax13_true = fig.add_subplot(2, 6, 10, projection='3d')
    true_function_values = np.array([f13_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax13_true, 'F13 True Function', x1_mesh, x2_mesh, true_function_values)
    # f16 - true
    ax16_true = fig.add_subplot(2, 6, 11, projection='3d')
    true_function_values = np.array([f16_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax16_true, 'F16 True Function', x1_mesh, x2_mesh, true_function_values)
    # f22 - true
    ax22_true = fig.add_subplot(2, 6, 12, projection='3d')
    true_function_values = np.array([f22_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)
    plot_3d_surface(ax22_true, 'F22 True Function', x1_mesh, x2_mesh, true_function_values)
    
    # show
    plt.show()

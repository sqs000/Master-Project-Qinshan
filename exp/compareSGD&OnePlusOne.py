import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
from nevergrad.optimization import optimizerlib
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)


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
    # true function values
    true_function_values = np.array([f_3_d_2_generator.problem(input) for input in input_mesh])
    true_function_values = true_function_values.reshape(x1_mesh.shape)


    # Plotting for SGD-optimized network
    fig_sgd = plt.figure(figsize=(15, 5))

    # create the NN with SGD optimization and plot its intial represented function
    sgd_network = hidden2_FNN(2, 50, 20, 1)
    ax1_sgd = fig_sgd.add_subplot(131, projection='3d')
    with torch.no_grad():
        sgd_initial_y = sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_initial_y = sgd_initial_y.reshape(x1_mesh.shape)
    plot_3d_surface(ax1_sgd, 'Initial Function (SGD)', x1_mesh, x2_mesh, sgd_initial_y)

    # training settings
    num_epochs = 1000
    sgd_learning_rate = 0.00005
    criterion = nn.MSELoss()
    sgd_optimizer = optim.SGD(sgd_network.parameters(), lr=sgd_learning_rate)

    # start training
    epochs = []
    epoch_losses = []
    for epoch in range(num_epochs):
        sgd_optimizer.zero_grad()
        # forward pass and evaluate
        predicted_y = sgd_network(data_x)
        epoch_loss = criterion(predicted_y, data_y)
        # backward pass and optimization
        epoch_loss.backward()
        sgd_optimizer.step()
        # record values for plotting
        epochs.append(epoch)
        epoch_losses.append(epoch_loss.item())
        # print the current best value at some intervals
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: MSE_Loss = {epoch_loss.item()}")

    # get the final best parameters and loss
    sgd_best_params = [param.data.tolist() for param in sgd_network.parameters()]
    def flatten_list(lst):
        result = []
        for el in lst:
            if isinstance(el, list):
                result.extend(flatten_list(el))
            else:
                result.append(el)
        return result
    sgd_best_params = np.array(flatten_list(sgd_best_params))
    with torch.no_grad():
        predicted_y = sgd_network(data_x)
        sgd_best_loss = criterion(predicted_y, data_y).item()

    # plot the final function learned with SGD
    ax2_sgd = fig_sgd.add_subplot(132, projection='3d')
    with torch.no_grad():
        sgd_predicted_y = sgd_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    sgd_predicted_y = sgd_predicted_y.reshape(x1_mesh.shape)
    plot_3d_surface(ax2_sgd, 'Function Learned with SGD', x1_mesh, x2_mesh, sgd_predicted_y)

    # plot the true function
    ax3_sgd = fig_sgd.add_subplot(133, projection='3d')
    plot_3d_surface(ax3_sgd, 'True Function', x1_mesh, x2_mesh, true_function_values)

    plt.suptitle('SGD-Optimized Network')




    # plotting for EA-optimized network
    fig_ea = plt.figure(figsize=(15, 5))

    # create the NN with EA optimization and plot its initial represented function
    ea_network = hidden2_FNN(2, 50, 20, 1)
    ax1_ea = fig_ea.add_subplot(131, projection='3d')
    with torch.no_grad():
        ea_initial_y = ea_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    ea_initial_y = ea_initial_y.reshape(x1_mesh.shape)
    plot_3d_surface(ax1_ea, 'Initial Function (EA)', x1_mesh, x2_mesh, ea_initial_y)

    # define objective function
    def objective_function(parameters):
        # assign the network with specific parameters
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in ea_network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(ea_network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))
        # evaluate the network with the parameters
        predicted_y = ea_network(data_x)
        criterion = nn.MSELoss()
        # return the loss to be minimized
        return criterion(predicted_y, data_y).item()

    # choose an optimizer
    num_parameters = sum(p.numel() for p in ea_network.parameters())
    ea_optimizer = optimizerlib.OnePlusOne(parametrization=num_parameters, budget=1000)

    # start training
    iterations = []
    objective_values = []
    for i in range(ea_optimizer.budget):
        # evolve
        recommendation = ea_optimizer.ask()
        objective_value = objective_function(recommendation.value)
        ea_optimizer.tell(recommendation, objective_value)
        # record values for plotting
        iterations.append(i)
        objective_values.append(objective_value)
        # print the current best value at some intervals
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}: MSE_Loss = {objective_function(ea_optimizer.provide_recommendation().value)}")

    # get the final best parameters and loss
    ea_best_params = ea_optimizer.provide_recommendation().value
    ea_best_loss = objective_function(ea_optimizer.provide_recommendation().value)

    # plot the final function learned with EA
    ax2_ea = fig_ea.add_subplot(132, projection='3d')
    with torch.no_grad():
        ea_predicted_y = ea_network(torch.tensor(input_mesh, dtype=torch.float32)).numpy()
    ea_predicted_y = ea_predicted_y.reshape(x1_mesh.shape)
    plot_3d_surface(ax2_ea, 'Function Learned with EA', x1_mesh, x2_mesh, ea_predicted_y)

    # plot the true function
    ax3_ea = fig_ea.add_subplot(133, projection='3d')
    plot_3d_surface(ax3_ea, 'True Function', x1_mesh, x2_mesh, true_function_values)

    plt.suptitle('EA-Optimized Network')


    # print final results
    print("SGD final Parameters:", sgd_best_params)
    print("SGD final Loss:", sgd_best_loss)
    print("EA final Parameters:", ea_best_params)
    print("EA final Loss:", ea_best_loss)

    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, label='SGD')
    plt.plot(iterations, objective_values, label='EA')
    plt.yscale('log')
    plt.title('SGD vs. EA optimization')
    plt.xlabel('Iteration/Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()


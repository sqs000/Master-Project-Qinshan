import sys
import os
# Get the current working directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to the sys.path
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
from nevergrad.optimization import optimizerlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)

    # set the number of runs to obtain multiple solutions
    num_runs = 10
    all_params = []
    all_losses = []
    # training
    for run in range(num_runs):
        # NN with EA optimization and random initialization
        ea_network = hidden2_FNN(2, 50, 20, 1)

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
            # return the loss
            return criterion(data_y, predicted_y).item()

        # choose an optimizer
        num_parameters = sum(p.numel() for p in ea_network.parameters())
        ea_optimizer = optimizerlib.OnePlusOne(parametrization=num_parameters, budget=5000)

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
            if i % 100 == 0:
                print(f"Iteration {i}: MSE_Loss = {objective_function(ea_optimizer.provide_recommendation().value)}")

        # Append the best parameters and loss for this run
        all_params.append(ea_optimizer.provide_recommendation().value)
        all_losses.append(objective_function(ea_optimizer.provide_recommendation().value))

    # Convert the lists to numpy arrays for easy manipulation
    all_params = np.array(all_params)
    all_losses = np.array(all_losses)

    # Calculate pairwise distances between parameter vectors
    distance_matrix = pdist(all_params, metric='euclidean')

    # Convert the distance matrix to a square matrix
    distance_matrix = squareform(distance_matrix)

    # Display the distance matrix
    print("Distance Matrix:")
    print(distance_matrix)

    # Increase diagonal values to be larger than the maximum distance for visualization
    distance_matrix[np.eye(num_runs, dtype=bool)] = np.max(distance_matrix) + 1.0

    # Visualize the distance matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix, cmap='viridis', annot=True, fmt=".2f", cbar_kws={'label': 'Euclidean Distance'})
    plt.title('Pairwise Distance Matrix between Solutions')
    plt.xlabel('Solution Index')
    plt.ylabel('Solution Index')

    # Visualize the final losses
    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.arange(num_runs), y=all_losses, palette='viridis')
    plt.title('Final Losses of EA-Optimized Networks')
    plt.xlabel('Solution Index')
    plt.ylabel('MSE Loss')

    plt.show()


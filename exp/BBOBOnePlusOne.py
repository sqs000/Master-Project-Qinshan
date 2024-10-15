import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
from data import data_generator
import torch
import torch.nn as nn
from network import hidden2_FNN
from nevergrad.optimization import optimizerlib
import numpy as np


if __name__ == "__main__":

    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000, standardize=False)

    n_repetitions = 20
    best_losses = []

    for r in range(n_repetitions):
        # network
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
        best_losses.append(ea_best_loss)
        print(f"Round {r} is finished.")

    print(f"OnePlusOne avg best loss: {np.mean(best_losses)}")

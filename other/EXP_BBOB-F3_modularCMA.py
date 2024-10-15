# from modcma import ModularCMAES
# cma = ModularCMAES(func, dim, budget=budget)

# while not cma.break_conditions():
#    cma.mutate()
#    cma.select()
#    cma.recombine()
#    cma.adapt()

import sys
import os
# Get the current working directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to the sys.path
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)

from modcma import AskTellCMAES
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np


if __name__ == "__main__":
        
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000, standardize=False)

    # repeated experiments
    n_repeatitions = 1
    # training settings
    num_epochs = 5000
    sgd_learning_rate = 0.0001
    criterion = nn.MSELoss()
    cma_budget = 5000
    # record processes
    sgd_training_losses = np.zeros((n_repeatitions, num_epochs))
    cma_training_losses = np.zeros((n_repeatitions, cma_budget))

    # start
    for round in range(n_repeatitions):

        # create the NN with SGD optimization
        sgd_network = hidden2_FNN(2, 50, 20, 1)
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
            # if (epoch+1) % 100 == 0:
            #     print(f"Epoch {epoch+1}: MSE_Loss = {epoch_loss.item()}")
        sgd_training_losses[round] = np.array(epoch_losses)


        # create the NN with EA optimization
        ea_network = hidden2_FNN(2, 50, 20, 1)
        num_parameters = sum(p.numel() for p in ea_network.parameters())
        cma = AskTellCMAES(d=num_parameters, budget=cma_budget, orthogonal=True, threshold_convergence=True, sample_sigma=True, mirrored="mirrored pairwise")

        # define the objective function
        def objective_function(parameters):
            # assign the network with specific parameters
            new_params = torch.split(torch.tensor(parameters), [p.numel() for p in ea_network.parameters()])
            with torch.no_grad():
                for param, new_param_value in zip(ea_network.parameters(), new_params):
                    param.data.copy_(new_param_value.reshape(param.data.shape))
            # evaluate the network with the parameters
            predicted_y = ea_network(data_x)
            # return the loss to be minimized
            return criterion(predicted_y, data_y).item()

        # start training
        iterations = []
        objective_values = []
        i = 0
        while not any(cma.break_conditions):
            # Retrieve a single new candidate solution
            recommendation = cma.ask()
            # Evaluate the objective function
            objective_value = objective_function(recommendation)
            # Update the algorithm with the objective function value
            cma.tell(recommendation, objective_value)
            iterations.append(i)
            objective_values.append(objective_value)
            i += 1
            
        cma_training_losses[round] = np.array(objective_values)

        print(f"Round {round+1} is finished.")

    avg_sgd_training_losses = np.mean(sgd_training_losses, axis=0)
    avg_cma_training_losses = np.mean(cma_training_losses, axis=0)
    epochs = np.arange(num_epochs)
    iterations = np.arange(cma_budget)
    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_sgd_training_losses, label='SGD')
    plt.plot(iterations, avg_cma_training_losses, label='CMA')
    plt.yscale('log')
    plt.title('SGD vs. CMA optimization')
    plt.xlabel('Iteration/Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()


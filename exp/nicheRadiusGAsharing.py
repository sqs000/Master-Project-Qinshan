import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import numpy as np
import matplotlib.pyplot as plt
from algorithm import GA_sharing


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)
    # calculate num_parameters
    opt_network = hidden2_FNN(2, 50, 20, 1)
    opt_network.to(torch.device("cpu"))
    num_parameters = sum(p.numel() for p in opt_network.parameters())
    def objective_function(parameters):
        """ Assign NN with parameters, calculate and return the loss. """
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(opt_network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))
        predicted_y = opt_network(data_x)
        criterion = nn.MSELoss()
        return criterion(predicted_y, data_y).item()
    # exp_settings
    n_repeatitions = 5
    budget_generations = 200
    niche_radius_list = [1, 5, 10, 20, 50, 100]
    set_0_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_1_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_2_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_3_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_4_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_5_training_losses = np.zeros((n_repeatitions, budget_generations))
    sets_training_losses = [set_0_training_losses, set_1_training_losses, set_2_training_losses, 
                            set_3_training_losses, set_4_training_losses, set_5_training_losses]
    # running
    for r in range(n_repeatitions):
        for i, radius in enumerate(niche_radius_list):
            final_pop, final_loss, generation_list, loss_list = GA_sharing(num_generations=budget_generations, population_size=1000, dim=num_parameters, p_m=0.04, niche_radius=radius, obj_f=objective_function, crossover_flag=True, crossover_type="param")
            sets_training_losses[i][r] = loss_list
            print(f"Set {i} is over.")
        print(f"Round {r} is over.")
    avg_set_0_training_losses = np.mean(set_0_training_losses, axis=0)
    avg_set_1_training_losses = np.mean(set_1_training_losses, axis=0)
    avg_set_2_training_losses = np.mean(set_2_training_losses, axis=0)
    avg_set_3_training_losses = np.mean(set_3_training_losses, axis=0)
    avg_set_4_training_losses = np.mean(set_4_training_losses, axis=0)
    avg_set_5_training_losses = np.mean(set_5_training_losses, axis=0)

    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(generation_list, avg_set_0_training_losses, label='1')
    plt.plot(generation_list, avg_set_1_training_losses, label='5')
    plt.plot(generation_list, avg_set_2_training_losses, label='10')
    plt.plot(generation_list, avg_set_3_training_losses, label='20')
    plt.plot(generation_list, avg_set_4_training_losses, label='50')
    plt.plot(generation_list, avg_set_5_training_losses, label='100')
    plt.yscale('log')
    plt.title('The NN training loss curve using GA optimization with fitness sharing (varying the niche radius)')
    plt.xlabel('Number of Generations')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()

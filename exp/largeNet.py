import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import hidden4_FNN
from data import data_generator
import numpy as np
import matplotlib.pyplot as plt
from algorithm import SGDBatchOpt, EvolveSGD, EvolveSGD_sharing
import random


n_repetitions = 3
lr = 0.000001
batch_size = 64

n_generations = 2
population_size = 200
n_epochs_GA_SGD = 2

n_epochs = 1000

SGD_training_losses = np.zeros((n_repetitions, n_generations))
GA_SGD_training_losses = np.zeros((n_repetitions, n_generations))
GA_SGD_sharing_R5_training_losses = np.zeros((n_repetitions, n_generations))
GA_SGD_sharing_R10_training_losses = np.zeros((n_repetitions, n_generations))


for i in range(n_repetitions):

    # indicate random seed
    np.random.seed(i+1)
    random.seed(i+1)
    torch.manual_seed(i+1)

    # network construction
    opt_network = hidden4_FNN(10, 64, 128, 64, 32, 1)
    opt_network.to(torch.device("cuda"))
    num_parameters = sum(p.numel() for p in opt_network.parameters())
    print([p.numel() for p in opt_network.parameters()], num_parameters)

    # data generation
    bbob_data_generator = data_generator(suite_name="bbob", function=3, dimension=10, instance=1, device=torch.device("cuda"))
    data_x, data_y = bbob_data_generator.generate(data_size=5000)

    # SGD
    criterion = nn.MSELoss()
    final_ind, final_loss, epochs, epoch_losses = SGDBatchOpt(opt_network, data_x, data_y, criterion, n_epochs, lr, batch_size)
    SGD_training_losses[i] = np.array(epoch_losses)[499::500]
    print("SGD is finished.")

    # network construction
    opt_network = hidden4_FNN(10, 64, 128, 64, 32, 1)
    opt_network.to(torch.device("cuda"))

    def objective_function(parameters):
        """ Assign NN with parameters, calculate and return the loss. """
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(opt_network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))
        predicted_y = opt_network(data_x)
        criterion = nn.MSELoss()
        return criterion(predicted_y, data_y).item()

    # GA_SGD
    final_pop_ga_sgd, final_loss_ga_sgd, generation_list, loss_list_ga_sgd = EvolveSGD(n_generations, population_size, num_parameters, objective_function, opt_network, data_x, data_y, nn.MSELoss(), n_epochs_GA_SGD, lr, batch_size)
    GA_SGD_training_losses[i] = np.array(loss_list_ga_sgd)
    print("GA_SGD is finished.")

    # GA_SGD_sharing
    opt_network = hidden4_FNN(10, 64, 128, 64, 32, 1)
    opt_network.to(torch.device("cuda"))
    niche_radius = 5
    final_pop_ga_sgd_sharing_5, final_loss_ga_sgd_sharing_5, generation_list_5, loss_list_ga_sgd_sharing_5 = EvolveSGD_sharing(n_generations, population_size, num_parameters, niche_radius, objective_function, opt_network, data_x, data_y, nn.MSELoss(), n_epochs_GA_SGD, lr, batch_size)
    GA_SGD_sharing_R5_training_losses[i] = np.array(loss_list_ga_sgd_sharing_5)
    print("GA_SGD_sharing_R5 is finished.")

    opt_network = hidden4_FNN(10, 64, 128, 64, 32, 1)
    opt_network.to(torch.device("cuda"))
    niche_radius = 10
    final_pop_ga_sgd_sharing_10, final_loss_ga_sgd_sharing_10, generation_list_10, loss_list_ga_sgd_sharing_10 = EvolveSGD_sharing(n_generations, population_size, num_parameters, niche_radius, objective_function, opt_network, data_x, data_y, nn.MSELoss(), n_epochs_GA_SGD, lr, batch_size)
    GA_SGD_sharing_R10_training_losses[i] = np.array(loss_list_ga_sgd_sharing_10)
    print("GA_SGD_sharing_R10 is finished.")

    print(f"Round {i} is finished.")

avg_SGD_training_losses = np.mean(SGD_training_losses, axis=0)
std_SGD_loss = np.std(SGD_training_losses, axis=0)

avg_GA_SGD_training_losses = np.mean(GA_SGD_training_losses, axis=0)
std_GA_SGD_loss = np.std(GA_SGD_training_losses, axis=0)

avg_GA_SGD_sharing_R5_training_losses = np.mean(GA_SGD_sharing_R5_training_losses, axis=0)
std_GA_SGD_sharing_R5_loss = np.std(GA_SGD_sharing_R5_training_losses, axis=0)

avg_GA_SGD_sharing_R10_training_losses = np.mean(GA_SGD_sharing_R10_training_losses, axis=0)
std_GA_SGD_sharing_R10_loss = np.std(GA_SGD_sharing_R10_training_losses, axis=0)

# plot the learning curve of loss
plt.tight_layout()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.tab10.colors))

x_range = range(1, 101)
plt.figure(figsize=(15, 10))

plt.plot(x_range, avg_SGD_training_losses, label='SGD')
plt.fill_between(x_range, avg_SGD_training_losses - std_SGD_loss, avg_SGD_training_losses + std_SGD_loss, alpha=0.2)

plt.plot(x_range, avg_GA_SGD_training_losses, label='GA_SGD')
plt.fill_between(x_range, avg_GA_SGD_training_losses - std_GA_SGD_loss, avg_GA_SGD_training_losses + std_GA_SGD_loss, alpha=0.2)

plt.plot(x_range, avg_GA_SGD_sharing_R5_training_losses, label='R=5')
plt.fill_between(x_range, avg_GA_SGD_sharing_R5_training_losses - std_GA_SGD_sharing_R5_loss, avg_GA_SGD_sharing_R5_training_losses + std_GA_SGD_sharing_R5_loss, alpha=0.2)

plt.plot(x_range, avg_GA_SGD_sharing_R10_training_losses, label='R=10')
plt.fill_between(x_range, avg_GA_SGD_sharing_R10_training_losses - std_GA_SGD_sharing_R10_loss, avg_GA_SGD_sharing_R10_training_losses + std_GA_SGD_sharing_R10_loss, alpha=0.2)

plt.yscale('log')
plt.title('SGD vs. GA_SGD vs. GA_SGD_sharing Optimization')
plt.xlabel('Number of evaluations (×5000000)')
plt.ylabel('MSE Loss (log scale)')
plt.legend()
plt.show()

import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import numpy as np
from data import data_generator
from network import hidden2_FNN
from algorithm import SGDOpt, GA, GA_sharing, GA_dynamic
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000, standardize=False)
    torch.save(data_x, "results/GA/data_x.pt")
    torch.save(data_y, "results/GA/data_y.pt")
    # experiment settings
    n_repetitions = 3
    budget_generations = 400
    sgd_lr = 0.00005
    criterion = nn.MSELoss()
    # result collection
    # loss
    sgd_losses = np.zeros((n_repetitions, budget_generations))
    ga_losses = np.zeros((n_repetitions, budget_generations))
    ga_sharing_losses = np.zeros((n_repetitions, budget_generations))
    ga_dynamic_losses = np.zeros((n_repetitions, budget_generations))
    # final population
    sgd_params = []
    ga_params = []
    ga_sharing_params = []
    ga_dynamic_params = []
    # final population evaluation
    sgd_results = []
    ga_results = []
    ga_sharing_results = []
    ga_dynamic_results = []
    # start running
    for r in range(n_repetitions):
        opt_network = hidden2_FNN(2, 50, 20, 1)
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
        final_ind, final_loss, epochs, epoch_losses = SGDOpt(opt_network, data_x, data_y, criterion, budget_generations, sgd_lr)
        final_pop_ga, final_loss_ga, generation_list, loss_list_ga = GA(num_generations=budget_generations, population_size=1000, dim=num_parameters, p_m=0.04, obj_f=objective_function, crossover_flag=True, crossover_type="param")
        final_pop_ga_sharing, final_loss_ga_sharing, generation_list, loss_list_ga_sharing = GA_sharing(num_generations=budget_generations, population_size=1000, dim=num_parameters, p_m=0.04, niche_radius=5, obj_f=objective_function, crossover_flag=True, crossover_type="param")
        final_pop_ga_dynamic, final_loss_ga_dynamic, generation_list, loss_list_ga_dynamic = GA_dynamic(num_generations=budget_generations, population_size=1000, dim=num_parameters, p_m=0.04, n_niches=50, niche_radius=5, obj_f=objective_function, crossover_flag=True, crossover_type="param")
        
        sgd_losses[r] = epoch_losses
        ga_losses[r] = loss_list_ga
        ga_sharing_losses[r] = loss_list_ga_sharing
        ga_dynamic_losses[r] = loss_list_ga_dynamic
        
        sgd_params.append(final_ind)
        ga_params.extend(final_pop_ga.tolist())
        ga_sharing_params.extend(final_pop_ga_sharing.tolist())
        ga_dynamic_params.extend(final_pop_ga_dynamic.tolist())
        
        sgd_results.append(final_loss)
        ga_results.extend(final_loss_ga)
        ga_sharing_results.extend(final_loss_ga_sharing)
        ga_dynamic_results.extend(final_loss_ga_dynamic)
        
        print(f"Round {r} is over.")

    np.save("results/GA/sgd_params", np.array(sgd_params))
    np.save("results/GA/ga_params", np.array(ga_params))
    np.save("results/GA/ga_sharing_params", np.array(ga_sharing_params))
    np.save("results/GA/ga_dynamic_params", np.array(ga_dynamic_params))
    
    np.save("results/GA/sgd_results", np.array(sgd_results))
    np.save("results/GA/ga_results", np.array(ga_results))
    np.save("results/GA/ga_sharing_results", np.array(ga_sharing_results))
    np.save("results/GA/ga_dynamic_results", np.array(ga_dynamic_results))

    avg_sgd_losses = np.mean(sgd_losses, axis=0)
    avg_ga_losses = np.mean(ga_losses, axis=0)
    avg_ga_sharing_losses = np.mean(ga_sharing_losses, axis=0)
    avg_ga_dynamic_losses = np.mean(ga_dynamic_losses, axis=0)
    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_sgd_losses, label='SGD')
    plt.plot(generation_list, avg_ga_losses, label='GA')
    plt.plot(generation_list, avg_ga_sharing_losses, label='GA_sharing')
    plt.plot(generation_list, avg_ga_dynamic_losses, label='GA_dynamic')
    plt.yscale('log')
    plt.title('The NN training loss curve using SGD, GA, GA_sharing or GA_dynamic optimization')
    plt.xlabel('Number of Generations / Epochs')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()


import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
from algorithm import GA_dynamic
from data import data_generator
from network import hidden2_FNN
import torch
import torch.nn as nn
import numpy as np


if __name__ == "__main__":
    # experiment settings
    n_repetitions = 1
    budget_generations = 1000
    ga_dynamic_function_losses = np.zeros((24, budget_generations))
    # start experiment running
    for function in range(1, 25, 1):
        # generate function data
        generator = data_generator(suite_name="bbob", function=function, dimension=2, instance=1, device=torch.device("cpu"))
        data_x, data_y = generator.generate(data_size=5000)
        # result collection
        ga_dynamic_losses = np.zeros((n_repetitions, budget_generations))
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
                return criterion(data_y, predicted_y).item()
            final_pop_ga_dynamic, final_loss_ga_dynamic, generation_list, loss_list_ga_dynamic = GA_dynamic(num_generations=budget_generations, population_size=1000, dim=num_parameters, p_m=0.04, n_niches=50, niche_radius=5, obj_f=objective_function, crossover_flag=True, crossover_type="param")
            ga_dynamic_losses[r] = loss_list_ga_dynamic
        avg_ga_dynamic_losses = np.mean(ga_dynamic_losses, axis=0)
        ga_dynamic_function_losses[function-1] = avg_ga_dynamic_losses
        print(f"Function {function} is finished with final averaged loss: {avg_ga_dynamic_losses[-1]}")
    # save
    np.save("results/GA/ga_dynamic_BBOB_F1~F24_losses", ga_dynamic_function_losses)
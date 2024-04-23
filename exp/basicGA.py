import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
from algorithm import GA


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)
    opt_network = hidden2_FNN(2, 50, 20, 1)
    opt_network.to(torch.device("cpu"))
    def objective_function(parameters):
        """ Assign NN with parameters, calculate and return the loss. """
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(opt_network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))
        predicted_y = opt_network(data_x)
        criterion = nn.MSELoss()
        return criterion(predicted_y, data_y).item()
    num_parameters = sum(p.numel() for p in opt_network.parameters())
    final_pop, final_loss, generation_list, loss_list = GA(num_generations=1000, population_size=1000, dim=num_parameters, p_m=0.04, obj_f=objective_function, crossover_flag=True, crossover_type="param")

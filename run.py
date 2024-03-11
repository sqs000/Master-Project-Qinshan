import argparse
from data import data_generator
from network import hidden2_FNN
from SGD import SGDOpt
from GA import genetic_algorithm as ga
from GA_sharing import genetic_algorithm as ga_sharing
from GA_dynamic import genetic_algorithm as ga_dynamic
import torch
import torch.nn as nn
import math
import numpy as np
import random

# argparse
parser = argparse.ArgumentParser(
            prog='run.py',
            description='Run NN optimization on an indicated BBOB function using an indicated algorithm.'
        )
# positional arguments
parser.add_argument('function', type=int, choices=range(1, 25), help='BBOB Function Number in [1, 25)')
parser.add_argument('algorithm', type=str, help='NN Optimization Algorithm in [SGD, GA, GA_sharing, GA_dynamic]')
parser.add_argument('numberofevaluations', type=int, help='The number of loss function evaluations during NN optimization')
# optional arguments
parser.add_argument('-l', '--learningrate', type=float, help='The learning rate for SGD')
parser.add_argument('-p', '--populationsize', type=int, help='The population size for GAs')
parser.add_argument('-m', '--mutationrate', type=float, help='The mutation rate for GAs')
parser.add_argument('-r', '--nicheradius', type=float, help='The niche radius for GA_sharing or GA_dynamic')
parser.add_argument('-n', '--numberofniches', type=int, help='The number of niches for GA_dynamic')
# required 
parser.add_argument('-i', '--instance', type=int, required=True, help='The random seed for the run')
# parse arguments
args = parser.parse_args()


# indicate random seed
INSTANCE = args.instance                                                                             
np.random.seed(INSTANCE)                                                                             
random.seed(INSTANCE)
# network construction
opt_network = hidden2_FNN(2, 50, 20, 1)
# data generation
bbob_data_generator = data_generator(suite_name="bbob", function=args.function, dimension=2, instance=1, device=torch.device("cpu"))
data_x, data_y = bbob_data_generator.generate(data_size=5000)
# run
if args.algorithm == "SGD":
    criterion = nn.MSELoss()
    budget_generations = args.numberofevaluations
    sgd_lr = args.learningrate
    final_ind, final_loss, epochs, epoch_losses = SGDOpt(opt_network, data_x, data_y, criterion, budget_generations, sgd_lr)
    print(f"The SGD final MSE loss after {budget_generations} epochs with lr {sgd_lr}: {final_loss}")
else:
    def objective_function(parameters):
        """ Assign NN with parameters, calculate and return the loss. """
        new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
        with torch.no_grad():
            for param, new_param_value in zip(opt_network.parameters(), new_params):
                param.data.copy_(new_param_value.reshape(param.data.shape))
        predicted_y = opt_network(data_x)
        criterion = nn.MSELoss()
        return criterion(data_y, predicted_y).item()
    population_size = args.populationsize
    budget_generations = math.ceil(args.numberofevaluations/population_size)
    num_parameters = sum(p.numel() for p in opt_network.parameters())
    p_m = args.mutationrate
    if args.algorithm == "GA":
        final_pop_ga, final_loss_ga, generation_list, loss_list_ga = ga(budget_generations, population_size, num_parameters, p_m, objective_function)
        print(f"The GA final MSE loss after {budget_generations} generations with popsize {population_size} and p_m {p_m}: {min(final_loss_ga)}")
    elif args.algorithm == "GA_sharing":
        niche_radius = args.nicheradius
        final_pop_ga_sharing, final_loss_ga_sharing, generation_list, loss_list_ga_sharing = ga_sharing(budget_generations, population_size, num_parameters, p_m, niche_radius, objective_function)
        print(f"The GA_sharing final MSE loss after {budget_generations} generations with popsize {population_size}, p_m {p_m} and niche_radius {niche_radius}: {min(final_loss_ga_sharing)}")
    elif args.algorithm == "GA_dynamic":
        n_niches = args.numberofniches
        niche_radius = args.nicheradius
        final_pop_ga_dynamic, final_loss_ga_dynamic, generation_list, loss_list_ga_dynamic = ga_dynamic(budget_generations, population_size, num_parameters, p_m, n_niches, niche_radius, objective_function)
        print(f"The GA_dynamic final MSE loss after {budget_generations} generations with popsize {population_size}, p_m {p_m}, n_niches {n_niches} and niche_radius {niche_radius}: {min(final_loss_ga_dynamic)}")
    else:
        print("Please select an optimization algorithm from this list: [SGD, GA, GA_sharing, GA_dynamic].")

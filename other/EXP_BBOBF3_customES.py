import sys
import os
# Get the current working directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to the sys.path
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import numpy as np
from data import data_generator
from network import hidden2_FNN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Initialization
def initialization(mu, dim):
    parent = np.zeros(shape=(mu, dim))
    init_sigma = np.ones(shape=(mu,))
    return parent, init_sigma


# Discrete recombination, use intermediate sigma
def recombination(population, parent_sigma, num_offsprings):
    offsprings = []
    offsprings_sigma = []
    for _ in range(num_offsprings):
        [p1, p2] = np.random.choice(len(population), 2, replace = False)
        choices = [population[p1], population[p2]]
        offspring = np.choose(np.random.randint(2, size=len(population[p1])), choices)
        offsprings.append(offspring)
        offsprings_sigma.append((parent_sigma[p1] + parent_sigma[p2])/2)
    return np.array(offsprings), np.array(offsprings_sigma)


# One-sigma mutation
def mutation(population, sigma, tau):
    mutated_population = np.copy(population)
    mutated_sigma = np.copy(sigma)
    mutated_sigma *= np.exp(np.random.normal(0, tau, len(sigma)))
    for individual, individual_sigma in zip(mutated_population, mutated_sigma):
        individual += np.random.normal(0, individual_sigma, len(individual))
    return mutated_population, mutated_sigma


# (mu + lambda) selection
def selection(parent_pop, offspring_pop, parent_sigma, offspring_sigma, obj_fun, mu):
    # mu + lambda
    population = np.concatenate((parent_pop, offspring_pop), axis=0)
    sigma = np.concatenate((parent_sigma, offspring_sigma))
    # evaluation and sort
    fitness_values = [obj_fun(x) for x in population]
    sorted_indices = np.argsort(fitness_values)
    sorted_fitness = [fitness_values[i] for i in sorted_indices]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_sigma = [sigma[i] for i in sorted_indices]
    # select
    selected_population = sorted_population[:mu]
    selected_sigma = sorted_sigma[:mu]
    return np.array(selected_population), np.array(selected_sigma), sorted_fitness


def customized_ES(dim, budget, mu_, lambda_, obj_fun):

    init_pop, init_sigma = initialization(mu_, dim)
    tau_0 =  1.0 / np.sqrt(dim)
    iterations = []
    objective_values = []
    best_obj_values = []
    n_evaluations = 0
    for i in range(budget):
        # recombination
        recombined_pop, recombined_sigma = recombination(init_pop, init_sigma, lambda_)
        # mutation
        mutated_pop, mutated_sigma = mutation(recombined_pop, recombined_sigma, tau_0)
        # selection
        selected_pop, selected_sigma, fitness_values = selection(init_pop, mutated_pop, init_sigma, mutated_sigma, obj_fun, mu_)
        n_evaluations += mu_+lambda_
        # reset
        init_pop = selected_pop
        init_sigma = selected_sigma
        # record
        iterations.append(i)
        objective_values.extend(fitness_values[::-1])
        best_obj_values.append(fitness_values[0])
        print(f"iteration {i}: {fitness_values[0]}")

    return init_pop, n_evaluations, objective_values, iterations, best_obj_values

    

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)

    # model construction
    ea_network = hidden2_FNN(2, 50, 20, 1)
    ea_network.to(device)
    num_parameters = sum(p.numel() for p in ea_network.parameters())

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
        return criterion(data_y, predicted_y).item()

    n_repeatitions = 10
    budget_iters = 500
    exp_sets = [[15, 200], [15, 300], [15, 500], [30, 200], [30, 300], [30, 500]]
    set_0_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_1_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_2_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_3_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_4_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_5_training_losses = np.zeros((n_repeatitions, budget_iters))
    sets_training_losses = [set_0_training_losses, set_1_training_losses, set_2_training_losses, set_3_training_losses, set_4_training_losses, set_5_training_losses]
    for r in range(n_repeatitions):
        for i, set in enumerate(exp_sets):
            population, eval_num, losses, iters, best_losses = customized_ES(dim=num_parameters, budget=budget_iters, mu_=set[0], lambda_=set[1], obj_fun=objective_function)
            sets_training_losses[i][r] = best_losses
        print(f"Round {r} is over.")
    avg_set_0_training_losses = np.mean(set_0_training_losses, axis=0)
    avg_set_1_training_losses = np.mean(set_1_training_losses, axis=0)
    avg_set_2_training_losses = np.mean(set_2_training_losses, axis=0)
    avg_set_3_training_losses = np.mean(set_3_training_losses, axis=0)
    avg_set_4_training_losses = np.mean(set_4_training_losses, axis=0)
    avg_set_5_training_losses = np.mean(set_5_training_losses, axis=0)

    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(iters, avg_set_0_training_losses, label='15+200')
    plt.plot(iters, avg_set_1_training_losses, label='15+300')
    plt.plot(iters, avg_set_2_training_losses, label='15+500')
    plt.plot(iters, avg_set_3_training_losses, label='30+200')
    plt.plot(iters, avg_set_4_training_losses, label='30+300')
    plt.plot(iters, avg_set_5_training_losses, label='30+500')
    plt.yscale('log')
    plt.title('Customized ES optimization changing mu+lambda')
    plt.xlabel('Number of Iterations')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()




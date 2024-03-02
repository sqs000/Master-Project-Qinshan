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
from utils import vector_euclidean_dist
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# sharing function
def sharing(distance, niche_radius, alpha_sh=1):
    if distance < niche_radius:
        return 1 - pow(distance/niche_radius, alpha_sh)
    else:
        return 0


# niche count
def niche_count(individual, population, niche_radius):
    count = 0
    for ind in population:
        dist = vector_euclidean_dist(ind, individual)
        count += sharing(dist, niche_radius)
    return count


# shared fitness
def f_sh(individual, population, obj_fun, niche_radius):
    f_ind = obj_fun(individual)
    m_ind = niche_count(individual, population, niche_radius)
    return m_ind * f_ind


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
def selection(parent_pop, offspring_pop, parent_sigma, offspring_sigma, obj_fun, mu, niche_radius):
    # mu + lambda
    population = np.concatenate((parent_pop, offspring_pop), axis=0)
    sigma = np.concatenate((parent_sigma, offspring_sigma))
    # evaluation and sort
    fitness_values = [f_sh(x, population, obj_fun, niche_radius) for x in population]
    sorted_indices = np.argsort(fitness_values)
    sorted_population = [population[i] for i in sorted_indices]
    sorted_sigma = [sigma[i] for i in sorted_indices]
    # select
    selected_population = sorted_population[:mu]
    selected_sigma = sorted_sigma[:mu]
    # real objective
    obj_func_values = [obj_fun(x) for x in population]
    sorted_indices_obj = np.argsort(obj_func_values)
    sorted_objective = [obj_func_values[i] for i in sorted_indices_obj]
    return np.array(selected_population), np.array(selected_sigma), np.array(sorted_objective)


def customized_ES(dim, budget, mu_, lambda_, obj_fun, niche_radius):

    init_pop, init_sigma = initialization(mu_, dim)
    tau_0 =  1.0 / np.sqrt(dim)
    iterations = []
    objective_values = []
    for i in range(budget):
        # recombination
        recombined_pop, recombined_sigma = recombination(init_pop, init_sigma, lambda_)
        # mutation
        mutated_pop, mutated_sigma = mutation(recombined_pop, recombined_sigma, tau_0)
        # selection
        selected_pop, selected_sigma, fitness_values = selection(init_pop, mutated_pop, init_sigma, mutated_sigma, obj_fun, mu_, niche_radius)
        # reset
        init_pop = selected_pop
        init_sigma = selected_sigma
        # record
        iterations.append(i)
        objective_values.append(fitness_values[0])
        # print(f"iteration {i}: {fitness_values[0]}")

    return init_pop, iterations, objective_values

    

if __name__ == "__main__":
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

    n_repeatitions = 5
    budget_iters = 500
    exp_sets = [0.1, 0.5, 1, 5, 10, 100]
    set_0_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_1_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_2_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_3_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_4_training_losses = np.zeros((n_repeatitions, budget_iters))
    set_5_training_losses = np.zeros((n_repeatitions, budget_iters))
    sets_training_losses = [set_0_training_losses, set_1_training_losses, set_2_training_losses, set_3_training_losses, set_4_training_losses, set_5_training_losses]
    for r in range(n_repeatitions):
        for i, set in enumerate(exp_sets):
            population, iters, best_losses = customized_ES(dim=num_parameters, budget=budget_iters, mu_=15, lambda_=300, obj_fun=objective_function, niche_radius=set)
            sets_training_losses[i][r] = best_losses
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
    plt.plot(iters, avg_set_0_training_losses, label='0.1')
    plt.plot(iters, avg_set_1_training_losses, label='0.5')
    plt.plot(iters, avg_set_2_training_losses, label='1')
    plt.plot(iters, avg_set_3_training_losses, label='5')
    plt.plot(iters, avg_set_4_training_losses, label='10')
    plt.plot(iters, avg_set_5_training_losses, label='100')
    plt.yscale('log')
    plt.title('Customized ES optimization changing the niche radius')
    plt.xlabel('Number of Iterations')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()




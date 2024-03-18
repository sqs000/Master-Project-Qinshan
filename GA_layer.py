import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Genetic Algorithm optimization of Neural Network parameters
def genetic_algorithm(num_generations, population_size, dim, p_m, obj_f):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness(population, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # Sexual reproduction with uniform crossover (without mutation)
        offspring = crossover(selected_parents)

        # Asexual reproduction with uniformly distributed mutation
        mutate(offspring, p_m)

        # Replace Old Population
        population, loss = replace_population(population, offspring, population_size, obj_f)

        print(f"Generation {generation} loss: {loss[0]}")
        generation_list.append(generation)
        loss_list.append(loss[0])
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list


def initialize_population(mu, dim):
    """ Initialize a population of mu individuals with dim dimensions. """
    population = np.random.normal(0, 1, (mu, dim))
    return population


def evaluate_fitness(population, obj_f):
    """ Evaluate fitness of each individual in the population. """
    loss_values = [obj_f(individual) for individual in population]
    fitness_values = [1 / loss if loss != 0 else float('inf') for loss in loss_values]
    return fitness_values


def roulette_wheel_selection_with_scaling(population, fitness_values):
    min_fitness = min(fitness_values)
    scaled_fitness = [fit - min_fitness for fit in fitness_values]
    total_scaled_fitness = sum(scaled_fitness)
    if total_scaled_fitness == 0:
        selection_probabilities = [1 / len(scaled_fitness) for fit in scaled_fitness]
    else:
        selection_probabilities = [fit / total_scaled_fitness for fit in scaled_fitness]
    selected_index = roulette_wheel_spin(selection_probabilities)
    selected_individual = population[selected_index]
    return selected_individual

def roulette_wheel_spin(probabilities):
    spin = random.uniform(0, 1)
    cumulative_probability = 0
    for i, prob in enumerate(probabilities):
        cumulative_probability += prob
        if spin <= cumulative_probability:
            return i

def select_parents(population, fitness_scores):
    """ Select parents based on fitness scores. """
    sorted_indices = np.argsort(fitness_scores)[::-1]
    sorted_fitness = [fitness_scores[i] for i in sorted_indices]
    sorted_population = [population[i] for i in sorted_indices]
    select_proportion = 0.4
    select_population = sorted_population[:int(select_proportion*len(sorted_population))]
    select_fitness = sorted_fitness[:int(select_proportion*len(sorted_fitness))]
    selected_parents = []
    for i in range(len(population)):
        selected_parents.append(roulette_wheel_selection_with_scaling(select_population, select_fitness))
    return selected_parents


def uniform_layer_crossover(parent1, parent2):
    assert len(parent1) == len(parent2), "Parents must have the same length"
    crossover_layer_masks = [random.choice([True, False]) for _ in range(3)]
    crossover_layer_n_params = [150, 1020, 21]
    child1 = []
    child2 = []
    start = 0
    for layer_mask, layer_n_params in zip(crossover_layer_masks, crossover_layer_n_params):
        end = start + layer_n_params
        if layer_mask:
            child1.extend(parent2[start:end])
            child2.extend(parent1[start:end])
        else:
            child1.extend(parent1[start:end])
            child2.extend(parent2[start:end])
        start += layer_n_params
    return child1, child2

def crossover(parents):
    """ Perform uniform crossover to create offspring. """
    offspring = []
    pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents)-(len(parents)%2), 2)]
    for pair in pairs:
        child1, child2 = uniform_layer_crossover(pair[0], pair[1])
        offspring.append(child1)
        offspring.append(child2)
    return np.array(offspring)


def mutate(offspring, mutation_rate):
    """ Apply mutation to introduce small random changes. """
    for individual in offspring:
        if random.uniform(0, 1) < mutation_rate:
            individual += np.random.uniform(-0.1, 0.1, len(individual))
        else:
            continue


def replace_population(old_population, new_population, mu, obj_f):
    """ Replace old population with new individuals. """
    population = np.concatenate((old_population, new_population), axis=0)
    # evaluation and sort
    loss_values = [obj_f(x) for x in population]
    sorted_indices = np.argsort(loss_values)
    sorted_loss = [loss_values[i] for i in sorted_indices]
    sorted_population = [population[i] for i in sorted_indices]
    # select
    selected_population = sorted_population[:mu]
    selected_loss = sorted_loss[:mu]
    return np.array(selected_population), selected_loss


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)
    opt_network = hidden2_FNN(2, 50, 20, 1)
    opt_network.to(device)
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
    final_pop, final_loss, generation_list, loss_list = genetic_algorithm(num_generations=1000, population_size=1000, dim=num_parameters, p_m=0.04, obj_f=objective_function)

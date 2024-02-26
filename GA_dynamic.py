import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import vector_euclidean_dist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# Genetic Algorithm optimization of Neural Network parameters
def genetic_algorithm(num_generations, population_size, dim, p_m, n_niches, niche_radius, obj_f):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness(population, n_niches, niche_radius, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # Sexual reproduction with uniform crossover (without mutation)
        offspring = crossover(selected_parents)

        # Asexual reproduction with uniformly distributed mutation
        mutate(offspring, p_m)

        # Replace Old Population
        population, loss = replace_population(population, offspring, population_size, n_niches, niche_radius, obj_f)
        best_loss = min(loss)

        # print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list


# Initialization
def initialize_population(mu, dim):
    """ Initialize a population of mu individuals with dim dimensions. """
    population = np.random.normal(0, 1, (mu, dim))
    return population


# Fitness function
def evaluate_fitness(population, n_niches, niche_radius, obj_f):
    """ Evaluate sharing fitness of each individual in the population. """
    loss_values = [obj_f(individual) for individual in population]
    fitness_values = [1 / loss if loss != 0 else float('inf') for loss in loss_values]
    dps = DPI(population, n_niches, niche_radius, obj_f)
    niche_sizes = size_niches(population, dps, niche_radius)
    sharing_fitness_values = [fitness/dynamic_niche_count(individual, population, niche_radius, dps, niche_sizes) for individual,fitness in zip(population,fitness_values)]
    return sharing_fitness_values

def dynamic_niche_count(individual, population, niche_radius, dps, niche_sizes):
    """ Dynamic niche count m_i^dyn to be divided by the original fitness. """
    peak_ind_flag = 0
    peak_index = 0
    for i, peak in enumerate(dps):
        if vector_euclidean_dist(individual, peak) < niche_radius:
            peak_ind_flag = 1
            peak_index = i
            break
    if peak_ind_flag:
        return niche_sizes[peak_index]
    else:
        count = 0
        for ind in population:
            dist = vector_euclidean_dist(ind, individual)
            count += sharing(dist, niche_radius)
        return count
    
def DPI(population, n_niches, niche_radius, obj_f):
    """ Dynamic peak identification. """
    losses = [obj_f(x) for x in population]
    sorted_indices = np.argsort(losses)
    sorted_population = [population[i] for i in sorted_indices]
    i = 0
    n_peaks = 0
    dps = []
    while n_peaks != n_niches and i < len(sorted_population):
        peak_flag = 1
        for peak in dps:
            if vector_euclidean_dist(sorted_population[i], peak) < niche_radius:
                peak_flag = 0
        if peak_flag:
            dps.append(sorted_population[i])
            n_peaks += 1
        i += 1
    return dps

def size_niches(population, dps, niche_radius):
    """ Return the niche sizes in the population corresponding to dps. """
    niche_sizes = np.zeros(len(dps), dtype=int)
    for ind in population:
        for i, peak in enumerate(dps):
            distance = vector_euclidean_dist(peak, ind)
            within_radius = distance < niche_radius
            niche_sizes[i] += int(within_radius)
    return niche_sizes.tolist()

def sharing(distance, niche_radius, alpha_sh=1):
    """ Sharing function. """
    if distance < niche_radius:
        return 1 - pow(distance/niche_radius, alpha_sh)
    else:
        return 0


# Parent Selection
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


# Crossover
def crossover(parents):
    """ Perform uniform crossover to create offspring. """
    offspring = []
    pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents)-(len(parents)%2), 2)]
    for pair in pairs:
        child1, child2 = uniform_crossover(pair[0], pair[1])
        offspring.append(child1)
        offspring.append(child2)
    return np.array(offspring)

def uniform_crossover(parent1, parent2):
    assert len(parent1) == len(parent2), "Parents must have the same length"
    crossover_mask = [random.choice([True, False]) for _ in range(len(parent1))]
    child1 = [p1 if mask else p2 for p1, p2, mask in zip(parent1, parent2, crossover_mask)]
    child2 = [p2 if mask else p1 for p1, p2, mask in zip(parent1, parent2, crossover_mask)]
    return child1, child2


# Mutation
def mutate(offspring, mutation_rate):
    """ Apply mutation to introduce small random changes. """
    for individual in offspring:
        if random.uniform(0, 1) < mutation_rate:
            individual += np.random.uniform(-0.1, 0.1, len(individual))
        else:
            continue


# Update Selection
def replace_population(old_population, new_population, mu, n_niches, niche_radius, obj_f):
    """ Replace old population with new individuals. """
    population = np.concatenate((old_population, new_population), axis=0)
    # evaluation and sort
    fitness_values = evaluate_fitness(population, n_niches, niche_radius, obj_f)
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_loss = [obj_f(ind) for ind in sorted_population]
    # select
    selected_population = sorted_population[:mu]
    selected_loss = sorted_loss[:mu]
    return np.array(selected_population), selected_loss


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)
    data_x.to(device=device)
    data_y.to(device=device)
    # calculate num_parameters
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
    # exp_settings
    n_repeatitions = 5
    budget_generations = 200
    n_niches_list = [5, 10, 20, 50, 100]
    set_0_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_1_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_2_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_3_training_losses = np.zeros((n_repeatitions, budget_generations))
    set_4_training_losses = np.zeros((n_repeatitions, budget_generations))
    sets_training_losses = [set_0_training_losses, set_1_training_losses, set_2_training_losses, 
                            set_3_training_losses, set_4_training_losses]
    # running
    for r in range(n_repeatitions):
        for i, n in enumerate(n_niches_list):
            final_pop, final_loss, generation_list, loss_list = genetic_algorithm(num_generations=budget_generations, population_size=1000, dim=num_parameters, p_m=0.04, n_niches=n, niche_radius=5, obj_f=objective_function)
            sets_training_losses[i][r] = loss_list
            print(f"Set {i} is over.")
        print(f"Round {r} is over.")
    avg_set_0_training_losses = np.mean(set_0_training_losses, axis=0)
    avg_set_1_training_losses = np.mean(set_1_training_losses, axis=0)
    avg_set_2_training_losses = np.mean(set_2_training_losses, axis=0)
    avg_set_3_training_losses = np.mean(set_3_training_losses, axis=0)
    avg_set_4_training_losses = np.mean(set_4_training_losses, axis=0)

    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(generation_list, avg_set_0_training_losses, label='5')
    plt.plot(generation_list, avg_set_1_training_losses, label='10')
    plt.plot(generation_list, avg_set_2_training_losses, label='20')
    plt.plot(generation_list, avg_set_3_training_losses, label='50')
    plt.plot(generation_list, avg_set_4_training_losses, label='100')
    plt.yscale('log')
    plt.title('The NN training loss curve using GA optimization with dynamic fitness sharing (varying the number of niches)')
    plt.xlabel('Number of Generations')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()

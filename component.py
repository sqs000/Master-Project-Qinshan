import numpy as np
import random
from sklearn.cluster import KMeans
from utils import vector_euclidean_dist
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from utils import assign_param, flatten_list


# Initialization
def initialize_population(mu, dim):
    """ Initialize a population of mu individuals with dim dimensions. """
    population = np.random.normal(0, 1, (mu, dim))
    return population


# GA Fitness function
def evaluate_fitness(population, obj_f):
    """ Evaluate fitness of each individual in the population. """
    loss_values = [obj_f(individual) for individual in population]
    fitness_values = [1 / loss if loss != 0 else float('inf') for loss in loss_values]
    return fitness_values

# GA_sharing Fitness function
def evaluate_fitness_sharing(population, niche_radius, obj_f):
    """ Evaluate sharing fitness of each individual in the population. """
    loss_values = [obj_f(individual) for individual in population]
    fitness_values = [1 / loss if loss != 0 else float('inf') for loss in loss_values]
    sharing_fitness_values = [fitness/niche_count(individual, population, niche_radius) for individual,fitness in zip(population,fitness_values)]
    return sharing_fitness_values

# GA_dynamic Fitness function
def evaluate_fitness_dynamic(population, n_niches, niche_radius, obj_f):
    """ Evaluate dynamic sharing fitness of each individual in the population. """
    loss_values = [obj_f(individual) for individual in population]
    fitness_values = [1 / loss if loss != 0 else float('inf') for loss in loss_values]
    dps = DPI(population, n_niches, niche_radius, obj_f)
    niche_sizes = size_niches(population, dps, niche_radius)
    sharing_fitness_values = [fitness/dynamic_niche_count(individual, population, niche_radius, dps, niche_sizes) for individual,fitness in zip(population,fitness_values)]
    return sharing_fitness_values

def niche_count(individual, population, niche_radius):
    """ Niche count m_i to be divided by the original fitness. """
    count = 0
    for ind in population:
        dist = vector_euclidean_dist(ind, individual)
        count += sharing(dist, niche_radius)
    return count

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
    
# GA_clustering fitness function
def evaluate_fitness_clustering(population, n_clusters, alpha, obj_f):
    """ Evaluate clustering fitness of each individual in the population. """
    loss_values = [obj_f(individual) for individual in population]
    fitness_values = [1 / loss if loss != 0 else float('inf') for loss in loss_values]
    clustering_fitness_values = [fitness/clustering_denominator(idx, individual, population, n_clusters, alpha) for idx,(individual,fitness) in enumerate(zip(population,fitness_values))]
    return clustering_fitness_values

def clustering_denominator(idx, individual, population, n_clusters, alpha):
    """ Computer the denominator of the clustering fitness. """
    est = KMeans(n_clusters=n_clusters)
    est.fit(population)
    centers = est.cluster_centers_
    labels = est.labels_
    label = labels[idx]
    center = centers[label]
    n_c = np.count_nonzero(labels == label)
    d_ic = vector_euclidean_dist(individual, center)
    d_max = compute_d_max(population, labels, label, center)
    return n_c * (1 - pow(d_ic/(2*d_max), alpha))

def compute_d_max(population, labels, label, center):
    """ Compute the max distance between an individual and its corresponding centroid in a specific niche. """
    distances = [vector_euclidean_dist(population[i], center) for i in range(len(population)) if labels[i] == label]
    return max(distances)



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
        selection_probabilities = [1 / len(scaled_fitness)] * len(scaled_fitness)
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
def crossover(parents, type):
    """ Perform uniform crossover on NN parameters, layers or nodes to create offspring. """
    offspring = []
    pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents)-(len(parents)%2), 2)]
    if type == "param":
        for pair in pairs:
            child1, child2 = uniform_crossover(pair[0], pair[1])
            offspring.append(child1)
            offspring.append(child2)
    elif type == "layer":
        for pair in pairs:
            child1, child2 = twopoint_layer_crossover(pair[0], pair[1])
            offspring.append(child1)
            offspring.append(child2)
    elif type == "node":
        for pair in pairs:
            child1, child2 = uniform_node_crossover(pair[0], pair[1])
            offspring.append(child1)
            offspring.append(child2)
    else:
        raise Exception("An error occurred. Incorrect crossover type.")
    return np.array(offspring)

def uniform_crossover(parent1, parent2):
    assert len(parent1) == len(parent2), "Parents must have the same length"
    crossover_mask = [random.choice([True, False]) for _ in range(len(parent1))]
    child1 = [p1 if mask else p2 for p1, p2, mask in zip(parent1, parent2, crossover_mask)]
    child2 = [p2 if mask else p1 for p1, p2, mask in zip(parent1, parent2, crossover_mask)]
    return child1, child2

def twopoint_layer_crossover(parent1, parent2):
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

def uniform_node_crossover(parent1, parent2):
    assert len(parent1) == len(parent2), "Parents must have the same length"
    parent1, parent2 = np.array(parent1), np.array(parent2)
    child1, child2 = np.zeros_like(parent1), np.zeros_like(parent2)
    # 50-node: 50*(2 + 1); 20-node: 20*(50 + 1); 1-node: 1*(20 + 1).
    indices_1 = [list(range(w,w+2))+[b] for w, b in zip(range(0, 100, 2), range(100, 150, 1))]
    indices_2 = [list(range(w,w+50))+[b] for w, b in zip(range(150, 1150, 50), range(1150, 1170, 1))]
    indices_3 = [list(range(w,w+20))+[b] for w, b in zip(range(1170, 1190, 20), range(1190, 1191, 1))]
    indices = indices_1 + indices_2 + indices_3
    crossover_node_masks = [random.choice([True, False]) for _ in range(50+20+1)]
    for index, crossover_mask in zip(indices, crossover_node_masks):
        if crossover_mask:
            child1[index] = parent2[index]
            child2[index] = parent1[index]
        else:
            child1[index] = parent1[index]
            child2[index] = parent2[index]
    return list(child1), list(child2)


# Mutation
def mutate(offspring, mutation_rate):
    """ Apply mutation to introduce small random changes. """
    for individual in offspring:
        if random.uniform(0, 1) < mutation_rate:
            individual += np.random.uniform(-0.1, 0.1, len(individual))
        else:
            continue


# Update Selection
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

# Fitness-Sharing Update Selection
def replace_population_sharing(old_population, new_population, mu, niche_radius, obj_f):
    """ Replace old population with new individuals. """
    population = np.concatenate((old_population, new_population), axis=0)
    # evaluation and sort
    fitness_values = evaluate_fitness_sharing(population, niche_radius, obj_f)
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_loss = [obj_f(ind) for ind in sorted_population]
    # select
    selected_population = sorted_population[:mu]
    selected_loss = sorted_loss[:mu]
    return np.array(selected_population), selected_loss

# Dynamic-Fitness-Sharing Update Selection
def replace_population_dynamic(old_population, new_population, mu, n_niches, niche_radius, obj_f):
    """ Replace old population with new individuals. """
    population = np.concatenate((old_population, new_population), axis=0)
    # evaluation and sort
    fitness_values = evaluate_fitness_dynamic(population, n_niches, niche_radius, obj_f)
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_loss = [obj_f(ind) for ind in sorted_population]
    # select
    selected_population = sorted_population[:mu]
    selected_loss = sorted_loss[:mu]
    return np.array(selected_population), selected_loss

# Clustering-Fitness Update Selection
def replace_population_clustering(old_population, new_population, mu, n_clusters, alpha, obj_f):
    """ Replace old population with new individuals. """
    population = np.concatenate((old_population, new_population), axis=0)
    # evaluation and sort
    fitness_values = evaluate_fitness_clustering(population, n_clusters, alpha, obj_f)
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_loss = [obj_f(ind) for ind in sorted_population]
    # select
    selected_population = sorted_population[:mu]
    selected_loss = sorted_loss[:mu]
    return np.array(selected_population), selected_loss


def sgd_step(population, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    """ Conduct SGD to the individuals in the population. """
    dataset = TensorDataset(data_x, data_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for individual in population:
        assign_param(network, individual)
        optimizer = optim.SGD(network.parameters(), lr=sgd_lr)
        for epoch in range(n_epochs):
            for batch_inputs, batch_targets in data_loader:
                optimizer.zero_grad()
                batch_outputs = network(batch_inputs)
                batch_loss = criterion(batch_outputs, batch_targets)
                batch_loss.backward()
                optimizer.step()
        final_params = [param.data.tolist() for param in network.parameters()]
        individual[:] = np.array(flatten_list(final_params))

import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import numpy as np
import math
from component import initialize_population, evaluate_fitness, select_parents, crossover, replace_population, evaluate_fitness_sharing, replace_population_sharing_separate


# Define the Rastrigin Function
def F5(x1, x2):
    x = np.array([x1, x2])
    N = len(x)
    c = 1.5
    return 10 * N + sum(x[i] ** 2 - 10 * math.cos(c * x[i]) for i in range(N))

# Define the Rastrigin Function
def GA_SGD_F5(x):
    N = len(x)
    c = 1.5
    return 10 * N + sum(x[i] ** 2 - 10 * math.cos(c * x[i]) for i in range(N))


# Define the gradient of the function F5
def gradient_F5(x1, x2):
    x = np.array([x1, x2])
    c = 1.5
    grad = np.zeros_like(x)
    grad[0] = 2 * x[0] + 10 * c * math.sin(c * x[0])
    grad[1] = 2 * x[1] + 10 * c * math.sin(c * x[1])
    return grad


# Stochastic Gradient Descent (SGD) to optimize F5
# optional parameter: tolerance=1e-6
def sgd_F5(initial_x1, initial_x2, learning_rate=0.001, max_iters=40000):
    x = np.array([initial_x1, initial_x2])
    for i in range(max_iters):
        grad = gradient_F5(x[0], x[1])
        x = x - learning_rate * grad
        # Check for convergence
        # if np.linalg.norm(grad) < tolerance:
        #     print(f'Convergence achieved at iteration {i}')
        #     break
    return x


# GA_SGD
def EvolveSGD(num_generations=100, population_size=200, dim=2, obj_f=GA_SGD_F5, n_epochs=2, sgd_lr=0.001):
    # Initialization
    # population = initialize_population(population_size, dim)
    population = np.full((population_size, dim), 10.0)
    generation_list = []
    f_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness(population, obj_f)
        # Parent Selection
        selected_parents = select_parents(population, fitness_scores)
        # SGD
        for i in range(len(selected_parents)):
            selected_parents[i] = sgd_F5(selected_parents[i][0], selected_parents[i][1], learning_rate=sgd_lr, max_iters=n_epochs)
        # Crossover
        offspring = crossover(selected_parents, type="param")
        # Replace Old Population
        population, pop_f_values = replace_population(population, offspring, population_size, obj_f)
        best_f = min(pop_f_values)
        # print(f"Generation {generation} function value: {best_loss}")
        generation_list.append(generation)
        f_list.append(best_f)
    # Final population contains optimized individuals
    return population, pop_f_values, generation_list, f_list


# GA_SGD
def EvolveSGD_sharing(num_generations=100, population_size=200, dim=2, obj_f=GA_SGD_F5, n_epochs=2, sgd_lr=0.001, niche_radius=10):
    # Initialization
    # population = initialize_population(population_size, dim)
    population = np.full((population_size, dim), 10.0)
    generation_list = []
    f_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness_sharing(population, niche_radius, obj_f)
        # Parent Selection
        selected_parents = select_parents(population, fitness_scores)
        # SGD
        for i in range(len(selected_parents)):
            selected_parents[i] = sgd_F5(selected_parents[i][0], selected_parents[i][1], learning_rate=sgd_lr, max_iters=n_epochs)
        # Crossover
        offspring = crossover(selected_parents, type="param")
        # Replace Old Population
        population, pop_f_values = replace_population_sharing_separate(population, offspring, population_size, niche_radius, obj_f)
        best_f = min(pop_f_values)
        # print(f"Generation {generation} function value: {best_loss}")
        generation_list.append(generation)
        f_list.append(best_f)
    # Final population contains optimized individuals
    return population, pop_f_values, generation_list, f_list


# Initial values for x1 and x2
# initial_x1 = 1.0
# initial_x2 = 1.0
initial_x1 = 10.0
initial_x2 = 10.0

# Run SGD
optimized_x = sgd_F5(initial_x1, initial_x2, learning_rate=0.5)
print(f'Optimized x1: {optimized_x[0]}, Optimized x2: {optimized_x[1]}, Function value: {F5(optimized_x[0], optimized_x[1])}')

# Run GA-SGD
population, pop_f_values, generation_list, f_list = EvolveSGD(sgd_lr=0.5)
print(f'Optimized x1: {population[0][0]}, Optimized x2: {population[0][1]}, Function value: {min(pop_f_values)}')

# Run GA-SGD-sharing
population, pop_f_values, generation_list, f_list = EvolveSGD_sharing(sgd_lr=0.5)
print(f'Optimized x1: {population[0][0]}, Optimized x2: {population[0][1]}, Function value: {min(pop_f_values)}')

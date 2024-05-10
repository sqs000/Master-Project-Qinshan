import torch
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from component import initialize_population, evaluate_fitness, evaluate_fitness_sharing, evaluate_fitness_dynamic,\
evaluate_fitness_clustering, select_parents, crossover, mutate, replace_population, replace_population_sharing,\
replace_population_dynamic, replace_population_clustering, replace_population_sharing_separate, replace_population_dynamic_separate, sgd_steps


def AdamBatchOpt(network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    dataset = TensorDataset(data_x, data_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=sgd_lr)
    epochs = []
    epoch_losses = []
    for epoch in range(n_epochs):
        loss = 0.0
        for batch_inputs, batch_targets in data_loader:
            optimizer.zero_grad()
            batch_outputs = network(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        epoch_loss = loss / len(data_loader)
        epochs.append(epoch)
        epoch_losses.append(epoch_loss)
    final_params = [param.data.tolist() for param in network.parameters()]
    def flatten_list(lst):
        result = []
        for el in lst:
            if isinstance(el, list):
                result.extend(flatten_list(el))
            else:
                result.append(el)
        return result
    final_params = np.array(flatten_list(final_params))
    with torch.no_grad():
        predicted_y = network(data_x)
        final_loss = criterion(predicted_y, data_y).item()
    return final_params, final_loss, epochs, epoch_losses


def SGDBatchOpt(network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    dataset = TensorDataset(data_x, data_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(network.parameters(), lr=sgd_lr)
    epochs = []
    epoch_losses = []
    for epoch in range(n_epochs):
        loss = 0.0
        for batch_inputs, batch_targets in data_loader:
            optimizer.zero_grad()
            batch_outputs = network(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        epoch_loss = loss / len(data_loader)
        epochs.append(epoch)
        epoch_losses.append(epoch_loss)
    final_params = [param.data.tolist() for param in network.parameters()]
    def flatten_list(lst):
        result = []
        for el in lst:
            if isinstance(el, list):
                result.extend(flatten_list(el))
            else:
                result.append(el)
        return result
    final_params = np.array(flatten_list(final_params))
    with torch.no_grad():
        predicted_y = network(data_x)
        final_loss = criterion(predicted_y, data_y).item()
    return final_params, final_loss, epochs, epoch_losses


def SGDOpt(network, data_x, data_y, criterion, n_epochs, sgd_lr):
    optimizer = optim.SGD(network.parameters(), lr=sgd_lr)
    epochs = []
    epoch_losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # forward pass and evaluate
        predicted_y = network(data_x)
        epoch_loss = criterion(predicted_y, data_y)
        # backward pass and optimization
        epoch_loss.backward()
        optimizer.step()
        # record values for plotting
        epochs.append(epoch)
        epoch_losses.append(epoch_loss.item())
    best_params = [param.data.tolist() for param in network.parameters()]
    def flatten_list(lst):
        result = []
        for el in lst:
            if isinstance(el, list):
                result.extend(flatten_list(el))
            else:
                result.append(el)
        return result
    best_params = np.array(flatten_list(best_params))
    return best_params, epoch_loss.item(), epochs, epoch_losses


def GA(num_generations, population_size, dim, p_m, obj_f, crossover_flag, crossover_type):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness(population, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # Sexual reproduction with uniform crossover
        if crossover_flag:
            offspring = crossover(selected_parents, crossover_type)
        else:
            offspring = selected_parents

        # Asexual reproduction with uniformly distributed mutation
        mutate(offspring, p_m)

        # Replace Old Population
        population, loss = replace_population(population, offspring, population_size, obj_f)

        # print(f"Generation {generation} loss: {loss[0]}")
        generation_list.append(generation)
        loss_list.append(loss[0])
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list


def GA_sharing(num_generations, population_size, dim, p_m, niche_radius, obj_f, crossover_flag, crossover_type):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness_sharing(population, niche_radius, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # Sexual reproduction with uniform crossover
        if crossover_flag:
            offspring = crossover(selected_parents, crossover_type)
        else:
            offspring = selected_parents

        # Asexual reproduction with uniformly distributed mutation
        mutate(offspring, p_m)

        # Replace Old Population
        population, loss = replace_population_sharing(population, offspring, population_size, niche_radius, obj_f)
        best_loss = min(loss)

        # print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list


def GA_dynamic(num_generations, population_size, dim, p_m, n_niches, niche_radius, obj_f, crossover_flag, crossover_type):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness_dynamic(population, n_niches, niche_radius, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # Sexual reproduction with uniform crossover
        if crossover_flag:
            offspring = crossover(selected_parents, crossover_type)
        else:
            offspring = selected_parents

        # Asexual reproduction with uniformly distributed mutation
        mutate(offspring, p_m)

        # Replace Old Population
        population, loss = replace_population_dynamic(population, offspring, population_size, n_niches, niche_radius, obj_f)
        best_loss = min(loss)

        # print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list


def GA_clustering(num_generations, population_size, dim, p_m, n_clusters, alpha, obj_f, crossover_flag, crossover_type):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness_clustering(population, n_clusters, alpha, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # Sexual reproduction with uniform crossover
        if crossover_flag:
            offspring = crossover(selected_parents, crossover_type)
        else:
            offspring = selected_parents

        # Asexual reproduction with uniformly distributed mutation
        mutate(offspring, p_m)

        # Replace Old Population
        population, loss = replace_population_clustering(population, offspring, population_size, n_clusters, alpha, obj_f)
        best_loss = min(loss)

        print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list


def EvolveSGD(num_generations, population_size, dim, obj_f, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness(population, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # SGD
        sgd_steps(selected_parents, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size)
        
        # Crossover
        offspring = crossover(selected_parents, type="param")

        # Replace Old Population
        population, loss = replace_population(population, offspring, population_size, obj_f)
        best_loss = min(loss)

        print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list

def EvolveSGD_sharing(num_generations, population_size, dim, niche_radius, obj_f, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness_sharing(population, niche_radius, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # SGD
        sgd_steps(selected_parents, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size)
        
        # Crossover
        offspring = crossover(selected_parents, type="param")

        # Replace Old Population
        population, loss = replace_population_sharing_separate(population, offspring, population_size, niche_radius, obj_f)
        best_loss = min(loss)

        print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list

def EvolveSGD_dynamic(num_generations, population_size, dim, n_niches, niche_radius, obj_f, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    # Initialization
    population = initialize_population(population_size, dim)
    generation_list = []
    loss_list = []
    for generation in range(num_generations):
        # Fitness Evaluation
        fitness_scores = evaluate_fitness_dynamic(population, n_niches, niche_radius, obj_f)

        # Selection
        selected_parents = select_parents(population, fitness_scores)

        # SGD
        sgd_steps(selected_parents, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size)
        
        # Crossover
        offspring = crossover(selected_parents, type="param")

        # Replace Old Population
        population, loss = replace_population_dynamic_separate(population, offspring, population_size, n_niches, niche_radius, obj_f)
        best_loss = min(loss)

        print(f"Generation {generation} loss: {best_loss}")
        generation_list.append(generation)
        loss_list.append(best_loss)
    # Final population contains optimized individuals
    return population, loss, generation_list, loss_list
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import numpy as np
import matplotlib.pyplot as plt
from algorithm import EvolveSGD_dynamic, SGDBatchOpt
import random
from utils import assign_param

# def EvolveSGD_1(num_generations, population_size, dim, obj_f, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
#     # Initialization
#     population = initialize_population(population_size, dim)
#     generation_list = []
#     loss_list = []
#     for generation in range(num_generations):
#         # Fitness Evaluation
#         # fitness_scores = evaluate_fitness(population, obj_f)

#         # Selection
#         # selected_parents = select_parents_evolveSGD(population, fitness_scores)

#         # Crossover
#         # offspring = crossover(selected_parents, type="param")
#         selected_parents = population
#         # SGD
#         sgd_step(selected_parents, network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size)
        
#         # Crossover
#         # offspring = crossover(selected_parents, type="param")

#         # Replace Old Population
#         population, loss = replace_population(population, selected_parents, population_size, obj_f)
#         best_loss = loss[0]

#         print(f"Generation {generation} loss: {best_loss}")
#         generation_list.append(generation)
#         loss_list.append(best_loss)
#     # Final population contains optimized individuals
#     return population, loss, generation_list, loss_list

# def select_parents_evolveSGD(population, fitness_scores):
#     mu, dim = population.shape
    
#     # Sort population indices based on fitness scores
#     sorted_indices = np.argsort(fitness_scores)
    
#     # Select parents iteratively until the size of selected population reaches mu
#     selected_parents = []
#     selected_indices = set()  # To ensure each parent is selected only once
    
#     while len(selected_parents) < mu:
#         # Randomly select the first parent from the first half of the sorted population
#         parent1_index = np.random.randint(mu // 2)
#         parent1 = population[sorted_indices[parent1_index]]
        
#         # Find the individual in the population with the longest Euclidean distance to parent1
#         max_distance = -1
#         max_distance_index = -1
        
#         for i in range(mu):
#             if i not in selected_indices:
#                 distance = np.linalg.norm(parent1 - population[i])
#                 if distance > max_distance:
#                     max_distance = distance
#                     max_distance_index = i
        
#         # Add the selected parents to the list
#         selected_parents.append(parent1)
#         selected_parents.append(population[max_distance_index])
        
#         # Add the indices of selected parents to the set
#         selected_indices.add(sorted_indices[parent1_index])
#         selected_indices.add(max_distance_index)
    
#     # Convert the list of selected parents to a 2D array
#     selected_parents = np.array(selected_parents)
    
#     return selected_parents

if __name__ == "__main__":
    # population, loss, generation_list, loss_list = EvolveSGD_dynamic(num_generations=100, population_size=200, dim=num_parameters, n_niches=20, niche_radius=5, obj_f=objective_function, network=opt_network, data_x=data_x, data_y=data_y, criterion=nn.MSELoss(), n_epochs=2, sgd_lr=0.00001, batch_size=64)
    # n_generation=100 popsize=200 n_epoch=2
    # way 0: n_generation*(popsize*n_epoch*10000 + popsize*5000)
    # way 0: 100 * (200*2*10000 + 200*5000) = 500000000
    # way 0: 500000000/10000 = 50000 epochs for SGD/Adam
    # way 1: n_generation*(popsize*n_epoch*5079 + popsize*5000)
    # way 1: 100 * (200*2*5079 + 200*5000) = 303160000
    # way 1: 303160000/5079 = 59689 epochs for SGD/Adam
    # 59689 - 50000 = 9689 epochs to be added
    # F3
    # ind1 = np.load("results_GA_SGD_sharing\F3\SGD\BBOB-3_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_GA_SGD_sharing\F3\SGD\BBOB-3_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_GA_SGD_sharing\F3\SGD\BBOB-3_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_GA_SGD_sharing\F3\SGD\BBOB-3_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_GA_SGD_sharing\F3\SGD\BBOB-3_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
    # F7
    # ind1 = np.load("results_GA_SGD_sharing\F7\SGD\BBOB-7_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_GA_SGD_sharing\F7\SGD\BBOB-7_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_GA_SGD_sharing\F7\SGD\BBOB-7_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_GA_SGD_sharing\F7\SGD\BBOB-7_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_GA_SGD_sharing\F7\SGD\BBOB-7_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
    # F13
    ind1 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    ind2 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    ind3 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    ind4 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    ind5 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
    nn1 = hidden2_FNN(2, 50, 20, 1)
    nn2 = hidden2_FNN(2, 50, 20, 1)
    nn3 = hidden2_FNN(2, 50, 20, 1)
    nn4 = hidden2_FNN(2, 50, 20, 1)
    nn5 = hidden2_FNN(2, 50, 20, 1)
    assign_param(nn1, ind1)
    assign_param(nn2, ind2)
    assign_param(nn3, ind3)
    assign_param(nn4, ind4)
    assign_param(nn5, ind5)
    networks = [nn1, nn2, nn3, nn4, nn5]
    criterion = nn.MSELoss()
    budget_epochs = 9689
    lr = 0.00001
    batch_size = 64
    for seed, network in zip(range(1, 6), networks):
        np.random.seed(seed)                                                                             
        random.seed(seed)
        torch.manual_seed(seed)
        # data generation
        generator = data_generator(suite_name="bbob", function=13, dimension=2, instance=1)
        data_x, data_y = generator.generate(data_size=5000)
        final_ind, final_loss, epochs, epoch_losses = SGDBatchOpt(network, data_x, data_y, criterion, budget_epochs, lr, batch_size)
        np.save('./results3/BBOB-13_SGD_newEval_50000-59689-epochs'+'_lr'+str(lr)+'_bs'+str(batch_size)+'_f-ind_i-'+str(seed), final_ind)
        np.save('./results3/BBOB-13_SGD_newEval_50000-59689-epochs'+'_lr'+str(lr)+'_bs'+str(batch_size)+'_f-loss_i-'+str(seed), np.array([final_loss]))
        np.save('./results3/BBOB-13_SGD_newEval_50000-59689-epochs'+'_lr'+str(lr)+'_bs'+str(batch_size)+'_losses_i-'+str(seed), np.array(epoch_losses))
        
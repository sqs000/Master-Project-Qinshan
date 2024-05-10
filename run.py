# import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
from data import data_generator
from network import hidden2_FNN
from algorithm import AdamBatchOpt, SGDBatchOpt, GA, GA_sharing, GA_dynamic, EvolveSGD, EvolveSGD_sharing, EvolveSGD_dynamic
import torch
# torch.set_num_threads(1)
import torch.nn as nn
import math
import numpy as np
import random


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
                prog='run.py',
                description='Run NN optimization on an indicated BBOB function using an indicated algorithm.'
            )
    # positional arguments
    parser.add_argument('--function', type=int, choices=range(1, 25), help='BBOB Function Number in [1, 25)')
    parser.add_argument('--algorithm', type=str, help='NN Optimization Algorithm in [Adam, SGD, GA, GA_sharing, GA_dynamic, GA_SGD, GA_SGD_sharing, GA_SGD_dynamic]')
    parser.add_argument('--numberofevaluations', type=int, help='The number of loss function evaluations during NN optimization')
    # optional arguments
    parser.add_argument('-l', '--learningrate', type=float, help='The learning rate for SGD or GA_SGDs')
    parser.add_argument('-b', '--batchsize', type=int, help='The batch size for SGD or GA_SGDs')
    parser.add_argument('-p', '--populationsize', type=int, help='The population size for GAs')
    parser.add_argument('-m', '--mutationrate', type=float, help='The mutation rate for GAs')
    parser.add_argument('-c', '--crossovertype', type=str, help='The type of crossover for GAs in [param, layer, node]. Disable crossover by using any other input')
    parser.add_argument('-r', '--nicheradius', type=float, help='The niche radius for GA_sharing, GA_SGD_sharing, GA_dynamic, GA_SGD_dynamic')
    parser.add_argument('-n', '--numberofniches', type=int, help='The number of niches for GA_dynamic or GA_SGD_dynamic')
    parser.add_argument('-e', '--numberofepochs', type=int, help='The number of epochs for GA_SGDs')
    # required 
    parser.add_argument('-i', '--instance', type=int, required=True, help='The random seed for the run')
    # parse arguments
    args = parser.parse_args()


    # indicate random seed
    INSTANCE = args.instance                                                                             
    np.random.seed(INSTANCE)                                                                             
    random.seed(INSTANCE)
    torch.manual_seed(INSTANCE)
    # network construction
    opt_network = hidden2_FNN(2, 50, 20, 1)
    opt_network.to(torch.device("cpu"))
    # data generation
    bbob_data_generator = data_generator(suite_name="bbob", function=args.function, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = bbob_data_generator.generate(data_size=5000)
    # run
    if args.algorithm == "SGD" or args.algorithm == "Adam":
        criterion = nn.MSELoss()
        budget_generations = math.ceil(args.numberofevaluations/10000)
        lr = args.learningrate
        batch_size = args.batchsize
        if args.algorithm == "SGD":
            final_ind, final_loss, epochs, epoch_losses = SGDBatchOpt(opt_network, data_x, data_y, criterion, budget_generations, lr, batch_size)
        else:
            final_ind, final_loss, epochs, epoch_losses = AdamBatchOpt(opt_network, data_x, data_y, criterion, budget_generations, lr, batch_size)
        np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-ind_i-'+str(INSTANCE), final_ind)
        np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-loss_i-'+str(INSTANCE), np.array([final_loss]))
        np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_losses_i-'+str(INSTANCE), np.array(epoch_losses))
    else:
        def objective_function(parameters):
            """ Assign NN with parameters, calculate and return the loss. """
            new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
            with torch.no_grad():
                for param, new_param_value in zip(opt_network.parameters(), new_params):
                    param.data.copy_(new_param_value.reshape(param.data.shape))
            predicted_y = opt_network(data_x)
            criterion = nn.MSELoss()
            return criterion(predicted_y, data_y).item()
        population_size = args.populationsize
        num_parameters = sum(p.numel() for p in opt_network.parameters())
        if "GA_SGD" not in args.algorithm:
            budget_generations = math.ceil((args.numberofevaluations-(population_size*5000))/(population_size*5000))
            if budget_generations < 1:
                budget_generations = 1
            p_m = args.mutationrate
            crossover_type = args.crossovertype
            crossover_flag = True if crossover_type in ["layer", "node", "param"] else False
        if args.algorithm == "GA":
            final_pop_ga, final_loss_ga, generation_list, loss_list_ga = GA(budget_generations, population_size, num_parameters, p_m, objective_function, crossover_flag, crossover_type)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_f-pop_i-'+str(INSTANCE), final_pop_ga)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_f-loss_i-'+str(INSTANCE), np.array(final_loss_ga))
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_losses_i-'+str(INSTANCE), np.array(loss_list_ga))
        elif args.algorithm == "GA_sharing":
            niche_radius = args.nicheradius
            final_pop_ga_sharing, final_loss_ga_sharing, generation_list, loss_list_ga_sharing = GA_sharing(budget_generations, population_size, num_parameters, p_m, niche_radius, objective_function, crossover_flag, crossover_type)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_R'+str(args.nicheradius)+'_f-pop_i-'+str(INSTANCE), final_pop_ga_sharing)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_R'+str(args.nicheradius)+'_f-loss_i-'+str(INSTANCE), np.array(final_loss_ga_sharing))
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_R'+str(args.nicheradius)+'_losses_i-'+str(INSTANCE), np.array(loss_list_ga_sharing))
        elif args.algorithm == "GA_dynamic":
            n_niches = args.numberofniches
            niche_radius = args.nicheradius
            final_pop_ga_dynamic, final_loss_ga_dynamic, generation_list, loss_list_ga_dynamic = GA_dynamic(budget_generations, population_size, num_parameters, p_m, n_niches, niche_radius, objective_function, crossover_flag, crossover_type)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_R'+str(args.nicheradius)+'_N'+str(args.numberofniches)+'_f-pop_i-'+str(INSTANCE), final_pop_ga_dynamic)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_R'+str(args.nicheradius)+'_N'+str(args.numberofniches)+'_f-loss_i-'+str(INSTANCE), np.array(final_loss_ga_dynamic))
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_M'+str(args.mutationrate)+'_C'+args.crossovertype+'_R'+str(args.nicheradius)+'_N'+str(args.numberofniches)+'_losses_i-'+str(INSTANCE), np.array(loss_list_ga_dynamic))
        elif args.algorithm == "GA_SGD":
            n_epochs = args.numberofepochs
            lr = args.learningrate
            batch_size = args.batchsize
            budget_generations = math.ceil( (args.numberofevaluations-(population_size*5000)) / (population_size*5000 + population_size*n_epochs*10000) )
            final_pop_ga_sgd, final_loss_ga_sgd, generation_list, loss_list_ga_sgd = EvolveSGD(budget_generations, population_size, num_parameters, objective_function, opt_network, data_x, data_y, nn.MSELoss(), n_epochs, lr, batch_size)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-pop_i-'+str(INSTANCE), final_pop_ga_sgd)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-loss_i-'+str(INSTANCE), np.array(final_loss_ga_sgd))
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_losses_i-'+str(INSTANCE), np.array(loss_list_ga_sgd))
        elif args.algorithm == "GA_SGD_sharing":
            n_epochs = args.numberofepochs
            lr = args.learningrate
            batch_size = args.batchsize
            niche_radius = args.nicheradius
            budget_generations = math.ceil( (args.numberofevaluations-(population_size*5000)) / (population_size*5000 + population_size*n_epochs*10000) )
            final_pop_ga_sgd_sharing, final_loss_ga_sgd_sharing, generation_list, loss_list_ga_sgd_sharing = EvolveSGD_sharing(budget_generations, population_size, num_parameters, niche_radius, objective_function, opt_network, data_x, data_y, nn.MSELoss(), n_epochs, lr, batch_size)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_R'+str(args.nicheradius)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-pop_i-'+str(INSTANCE), final_pop_ga_sgd_sharing)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_R'+str(args.nicheradius)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-loss_i-'+str(INSTANCE), np.array(final_loss_ga_sgd_sharing))
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_R'+str(args.nicheradius)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_losses_i-'+str(INSTANCE), np.array(loss_list_ga_sgd_sharing))
        elif args.algorithm == "GA_SGD_dynamic":
            n_epochs = args.numberofepochs
            lr = args.learningrate
            batch_size = args.batchsize
            n_niches = args.numberofniches
            niche_radius = args.nicheradius
            budget_generations = math.ceil( (args.numberofevaluations-(population_size*5000)) / (population_size*5000 + population_size*n_epochs*10000) )
            final_pop_ga_sgd_dynamic, final_loss_ga_sgd_dynamic, generation_list, loss_list_ga_sgd_dynamic = EvolveSGD_dynamic(budget_generations, population_size, num_parameters, n_niches, niche_radius, objective_function, opt_network, data_x, data_y, nn.MSELoss(), n_epochs, lr, batch_size)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_R'+str(args.nicheradius)+'_N'+str(args.numberofniches)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-pop_i-'+str(INSTANCE), final_pop_ga_sgd_dynamic)
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_R'+str(args.nicheradius)+'_N'+str(args.numberofniches)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_f-loss_i-'+str(INSTANCE), np.array(final_loss_ga_sgd_dynamic))
            np.save('./results3/BBOB-'+str(args.function)+'_'+args.algorithm+'_E'+str(args.numberofevaluations)+'_P'+str(args.populationsize)+'_R'+str(args.nicheradius)+'_N'+str(args.numberofniches)+'_ne'+str(args.numberofepochs)+'_lr'+str(args.learningrate)+'_bs'+str(args.batchsize)+'_losses_i-'+str(INSTANCE), np.array(loss_list_ga_sgd_dynamic))
        else:
            print("Please select an optimization algorithm from this list: [Adam, SGD, GA, GA_sharing, GA_dynamic, GA_SGD, GA_SGD_sharing, GA_SGD_dynamic].")

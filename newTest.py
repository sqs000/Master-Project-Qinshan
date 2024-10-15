import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import numpy as np
import random


if __name__ == "__main__":

    # load results
    # GA_SGD_sharing
    # F1: R=10 ×
    # pop1 = np.load("results_GA_SGD_sharing\F1\R=10\BBOB-1_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F1\R=10\BBOB-1_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F1\R=10\BBOB-1_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F1\R=10\BBOB-1_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F1\R=10\BBOB-1_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F3: R=5 √ √
    # pop1 = np.load("results_GA_SGD_sharing\F3\R=5\BBOB-3_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F3\R=5\BBOB-3_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F3\R=5\BBOB-3_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F3\R=5\BBOB-3_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F3\R=5\BBOB-3_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F3: GA-SGD
    # pop1 = np.load("results_GA_SGD_sharing\F3\GA_SGD\BBOB-3_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F3\GA_SGD\BBOB-3_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F3\GA_SGD\BBOB-3_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F3\GA_SGD\BBOB-3_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F3\GA_SGD\BBOB-3_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F7: GA-SGD √ √
    # pop1 = np.load("results_GA_SGD_sharing\F7\GA_SGD\BBOB-7_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F7\GA_SGD\BBOB-7_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F7\GA_SGD\BBOB-7_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F7\GA_SGD\BBOB-7_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F7\GA_SGD\BBOB-7_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F7: GA-SGD_sharing R=20
    # pop1 = np.load("results_GA_SGD_sharing\F7\R=20\BBOB-7_GA_SGD_sharing_E500000000_P200_R20.0_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F7\R=20\BBOB-7_GA_SGD_sharing_E500000000_P200_R20.0_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F7\R=20\BBOB-7_GA_SGD_sharing_E500000000_P200_R20.0_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F7\R=20\BBOB-7_GA_SGD_sharing_E500000000_P200_R20.0_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F7\R=20\BBOB-7_GA_SGD_sharing_E500000000_P200_R20.0_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F13: R=5 √ √
    pop1 = np.load("results_GA_SGD_sharing\F13\R=5\BBOB-13_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    pop2 = np.load("results_GA_SGD_sharing\F13\R=5\BBOB-13_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    pop3 = np.load("results_GA_SGD_sharing\F13\R=5\BBOB-13_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    pop4 = np.load("results_GA_SGD_sharing\F13\R=5\BBOB-13_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    pop5 = np.load("results_GA_SGD_sharing\F13\R=5\BBOB-13_GA_SGD_sharing_E500000000_P200_R5.0_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F13: GA-SGD
    # pop1 = np.load("results_GA_SGD_sharing\F13\GA_SGD\BBOB-13_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F13\GA_SGD\BBOB-13_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F13\GA_SGD\BBOB-13_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F13\GA_SGD\BBOB-13_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F13\GA_SGD\BBOB-13_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F16: GA-SGD ×
    # pop1 = np.load("results_GA_SGD_sharing\F16\GA_SGD\BBOB-16_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F16\GA_SGD\BBOB-16_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F16\GA_SGD\BBOB-16_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F16\GA_SGD\BBOB-16_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F16\GA_SGD\BBOB-16_GA_SGD_E500000000_P200_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    # F22: R=10 ×
    # pop1 = np.load("results_GA_SGD_sharing\F22\R=10\BBOB-22_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_sharing\F22\R=10\BBOB-22_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_sharing\F22\R=10\BBOB-22_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_sharing\F22\R=10\BBOB-22_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_sharing\F22\R=10\BBOB-22_GA_SGD_sharing_E500000000_P200_R10.0_ne2_lr1e-05_bs64_f-pop_i-5.npy")
    
    # GA_SGD_dynamic 
    # F3: R=5 N=20
    # pop1 = np.load("results_GA_SGD_dynamic\F3\R=5 N=20\BBOB-3_GA_SGD_dynamic_E500000000_P200_R5.0_N20_ne2_lr1e-05_bs64_f-pop_i-1.npy")
    # pop2 = np.load("results_GA_SGD_dynamic\F3\R=5 N=20\BBOB-3_GA_SGD_dynamic_E500000000_P200_R5.0_N20_ne2_lr1e-05_bs64_f-pop_i-2.npy")
    # pop3 = np.load("results_GA_SGD_dynamic\F3\R=5 N=20\BBOB-3_GA_SGD_dynamic_E500000000_P200_R5.0_N20_ne2_lr1e-05_bs64_f-pop_i-3.npy")
    # pop4 = np.load("results_GA_SGD_dynamic\F3\R=5 N=20\BBOB-3_GA_SGD_dynamic_E500000000_P200_R5.0_N20_ne2_lr1e-05_bs64_f-pop_i-4.npy")
    # pop5 = np.load("results_GA_SGD_dynamic\F3\R=5 N=20\BBOB-3_GA_SGD_dynamic_E500000000_P200_R5.0_N20_ne2_lr1e-05_bs64_f-pop_i-5.npy")

    # SGD oldEval (50000 epochs)
    # F1
    # ind1 = np.load("results_GA_SGD_sharing\F1\SGD\BBOB-1_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_GA_SGD_sharing\F1\SGD\BBOB-1_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_GA_SGD_sharing\F1\SGD\BBOB-1_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_GA_SGD_sharing\F1\SGD\BBOB-1_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_GA_SGD_sharing\F1\SGD\BBOB-1_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
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
    # ind1 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_GA_SGD_sharing\F13\SGD\BBOB-13_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
    # F16
    # ind1 = np.load("results_GA_SGD_sharing\F16\SGD\BBOB-16_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_GA_SGD_sharing\F16\SGD\BBOB-16_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_GA_SGD_sharing\F16\SGD\BBOB-16_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_GA_SGD_sharing\F16\SGD\BBOB-16_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_GA_SGD_sharing\F16\SGD\BBOB-16_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
    # F22
    # ind1 = np.load("results_GA_SGD_sharing\F22\SGD\BBOB-22_SGD_E500000000_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_GA_SGD_sharing\F22\SGD\BBOB-22_SGD_E500000000_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_GA_SGD_sharing\F22\SGD\BBOB-22_SGD_E500000000_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_GA_SGD_sharing\F22\SGD\BBOB-22_SGD_E500000000_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_GA_SGD_sharing\F22\SGD\BBOB-22_SGD_E500000000_lr1e-05_bs64_f-ind_i-5.npy")
    
    # SGD newEval (59689 epochs)
    # F3
    # ind1 = np.load("results_SGD_newEval\F3\BBOB-3_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_SGD_newEval\F3\BBOB-3_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_SGD_newEval\F3\BBOB-3_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_SGD_newEval\F3\BBOB-3_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_SGD_newEval\F3\BBOB-3_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-5.npy")
    # F7
    # ind1 = np.load("results_SGD_newEval\F7\BBOB-7_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-1.npy")
    # ind2 = np.load("results_SGD_newEval\F7\BBOB-7_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-2.npy")
    # ind3 = np.load("results_SGD_newEval\F7\BBOB-7_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-3.npy")
    # ind4 = np.load("results_SGD_newEval\F7\BBOB-7_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-4.npy")
    # ind5 = np.load("results_SGD_newEval\F7\BBOB-7_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-5.npy")
    # F13
    ind1 = np.load("results_SGD_newEval\F13\BBOB-13_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-1.npy")
    ind2 = np.load("results_SGD_newEval\F13\BBOB-13_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-2.npy")
    ind3 = np.load("results_SGD_newEval\F13\BBOB-13_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-3.npy")
    ind4 = np.load("results_SGD_newEval\F13\BBOB-13_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-4.npy")
    ind5 = np.load("results_SGD_newEval\F13\BBOB-13_SGD_newEval_50000-59689-epochs_lr1e-05_bs64_f-ind_i-5.npy")

    GA_SGD_losses = []
    SGD_losses = []
    
    generator = data_generator(suite_name="bbob", function=13, dimension=2, instance=1)

    for seed in range(6, 106):
        
        # test data generation
        np.random.seed(seed)                                                                             
        random.seed(seed)
        torch.manual_seed(seed)
        data_x, data_y = generator.generate(data_size=5000)
        
        # objective function
        opt_network = hidden2_FNN(2, 50, 20, 1)
        def objective_function(parameters):
            """ Assign NN with parameters, calculate and return the loss. """
            new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
            with torch.no_grad():
                for param, new_param_value in zip(opt_network.parameters(), new_params):
                    param.data.copy_(new_param_value.reshape(param.data.shape))
            predicted_y = opt_network(data_x)
            criterion = nn.MSELoss()
            return criterion(predicted_y, data_y).item()
        
        GA_SGD_population = np.concatenate((pop1[0], pop2[0], pop3[0], pop4[0], pop5[0]), axis=0).reshape((5, -1))
        SGD_population = np.concatenate((ind1, ind2, ind3, ind4, ind5), axis=0).reshape((5, -1))
        GA_SGD_losses = [objective_function(x) for x in GA_SGD_population]
        SGD_losses = [objective_function(x) for x in SGD_population]
        GA_SGD_losses.append(np.mean(GA_SGD_losses))
        SGD_losses.append(np.mean(SGD_losses))

    print(f"GA_SGD_mean_loss: {np.mean(GA_SGD_losses)}.")
    print(f"SGD_mean_loss: {np.mean(SGD_losses)}.")

    print(f"GA_SGD_best_loss: {np.min(GA_SGD_losses)}.")
    print(f"SGD_best_loss: {np.min(SGD_losses)}.")

    print(f"GA_SGD_worst_loss: {np.max(GA_SGD_losses)}.")
    print(f"SGD_worst_loss: {np.max(SGD_losses)}.")

    print(f"GA_SGD_std_deviation: {np.std(GA_SGD_losses, ddof=1)}.")
    print(f"SGD_std_deviation: {np.std(SGD_losses, ddof=1)}.")

    print(f"GA_SGD_std_error: {np.std(GA_SGD_losses, ddof=1)/np.sqrt(len(GA_SGD_losses))}.")
    print(f"SGD_std_error: {np.std(SGD_losses, ddof=1)/np.sqrt(len(SGD_losses))}.")

    print(f"GA_SGD_1Q: {np.percentile(GA_SGD_losses, 25)}.")
    print(f"SGD_1Q: {np.percentile(SGD_losses, 25)}.")

    print(f"GA_SGD_median: {np.percentile(GA_SGD_losses, 50)}.")
    print(f"SGD_median: {np.percentile(SGD_losses, 50)}.")

    print(f"GA_SGD_3Q: {np.percentile(GA_SGD_losses, 75)}.")
    print(f"SGD_3Q: {np.percentile(SGD_losses, 75)}.")

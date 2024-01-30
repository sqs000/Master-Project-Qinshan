import numpy as np
from data import data_generator
from network import hidden2_FNN
from EXP_BBOBF3_customES import customized_ES as ES
from EXP_BBOBF3_customES_sharing import customized_ES as ES_sharing
from EXP_BBOBF3_customES_dynamic import customized_ES as ES_dynamic
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)

    # model construction
    ea_network = hidden2_FNN(2, 50, 20, 1)
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
    ES_losses = np.zeros((n_repeatitions, budget_iters))
    ES_sharing_losses = np.zeros((n_repeatitions, budget_iters))
    ES_dynamic_losses = np.zeros((n_repeatitions, budget_iters))
    
    for r in range(n_repeatitions):
        population, eval_num, losses, iters, best_losses_ES = ES(dim=num_parameters, budget=budget_iters, mu_=15, lambda_=300, obj_fun=objective_function)
        population, iters, best_losses_ES_sharing = ES_sharing(dim=num_parameters, budget=budget_iters, mu_=15, lambda_=300, obj_fun=objective_function, niche_radius=1)
        population, iters, best_losses_ES_dynamic = ES_dynamic(dim=num_parameters, budget=budget_iters, mu_=15, lambda_=300, obj_fun=objective_function, n_niches=10, niche_radius=1)
        ES_losses[r] = best_losses_ES
        ES_sharing_losses[r] = best_losses_ES_sharing
        ES_dynamic_losses[r] = best_losses_ES_dynamic
        print(f"Round {r} is over.")
    
    ES_final_losses = ES_losses[:, -1]
    ES_sharing_final_losses = ES_sharing_losses[:, -1]
    ES_dynamic_final_losses = ES_dynamic_losses[:, -1]

    mean_loss = {
                    "ES": 0, 
                    "ES_sharing": 0, 
                    "ES_dynamic": 0
                }
    std_deviation = {
                    "ES": 0, 
                    "ES_sharing": 0, 
                    "ES_dynamic": 0
                    }
    std_error = {
                    "ES": 0, 
                    "ES_sharing": 0, 
                    "ES_dynamic": 0
                }
    oneQ = {
                "ES": 0, 
                "ES_sharing": 0, 
                "ES_dynamic": 0
            }
    median = {
                "ES": 0, 
                "ES_sharing": 0, 
                "ES_dynamic": 0
            }
    threeQ = {
                "ES": 0, 
                "ES_sharing": 0, 
                "ES_dynamic": 0
            }
    best_loss = {
                    "ES": 0, 
                    "ES_sharing": 0, 
                    "ES_dynamic": 0
                }
    mean_loss["ES"] = np.mean(ES_final_losses)
    mean_loss["ES_sharing"] = np.mean(ES_sharing_final_losses)
    mean_loss["ES_dynamic"] = np.mean(ES_dynamic_final_losses)
    std_deviation["ES"] = np.std(ES_final_losses, ddof=1)
    std_deviation["ES_sharing"] = np.std(ES_sharing_final_losses, ddof=1)
    std_deviation["ES_dynamic"] = np.std(ES_dynamic_final_losses, ddof=1)
    std_error["ES"] = np.std(ES_final_losses, ddof=1) / np.sqrt(len(ES_final_losses))
    std_error["ES_sharing"] = np.std(ES_sharing_final_losses, ddof=1) / np.sqrt(len(ES_sharing_final_losses))
    std_error["ES_dynamic"] = np.std(ES_dynamic_final_losses, ddof=1) / np.sqrt(len(ES_dynamic_final_losses))
    oneQ["ES"] = np.percentile(ES_final_losses, 25)
    oneQ["ES_sharing"] = np.percentile(ES_sharing_final_losses, 25)
    oneQ["ES_dynamic"] = np.percentile(ES_dynamic_final_losses, 25)
    median["ES"] = np.percentile(ES_final_losses, 50)
    median["ES_sharing"] = np.percentile(ES_sharing_final_losses, 50)
    median["ES_dynamic"] = np.percentile(ES_dynamic_final_losses, 50)
    threeQ["ES"] = np.percentile(ES_final_losses, 75)
    threeQ["ES_sharing"] = np.percentile(ES_sharing_final_losses, 75)
    threeQ["ES_dynamic"] = np.percentile(ES_dynamic_final_losses, 75)
    best_loss["ES"] = np.max(ES_final_losses)
    best_loss["ES_sharing"] = np.max(ES_sharing_final_losses)
    best_loss["ES_dynamic"] = np.max(ES_dynamic_final_losses)

    # List of dictionaries
    list_of_dicts = [mean_loss, std_deviation, std_error, oneQ, median, threeQ, best_loss]

    # Iterate through the list and print each dictionary
    for idx, d in enumerate(list_of_dicts, start=1):
        print(f"Dictionary {idx}:")
        for key, value in d.items():
            print(f"  {key}: {value}")
        print()

    avg_ES_losses = np.mean(ES_losses, axis=0)
    avg_ES_sharing_losses = np.mean(ES_sharing_losses, axis=0)
    avg_ES_dynamic_losses = np.mean(ES_dynamic_losses, axis=0)
    # plot the learning curve of loss
    plt.figure(figsize=(10, 6))
    plt.plot(iters, avg_ES_losses, label='ES')
    plt.plot(iters, avg_ES_sharing_losses, label='ES_sharing')
    plt.plot(iters, avg_ES_dynamic_losses, label='ES_dynamic')
    plt.yscale('log')
    plt.title('Add Fitness Sharing or Dynamic Fitness Sharing to a customized ES Optimization')
    plt.xlabel('Number of Iterations')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()


import numpy as np
from GA_dynamic import DPI
from network import hidden2_FNN
import torch
import torch.nn as nn
from utils import vector_euclidean_dist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
data_x = torch.load("results/GA/data_x.pt")
data_y = torch.load("results/GA/data_y.pt")
data_x.to(device)
data_y.to(device)

# build network
opt_network = hidden2_FNN(2, 50, 20, 1)
opt_network.to(device)

# define objective function to rank individuals
def objective_function(parameters):
    """ Assign NN with parameters, calculate and return the loss. """
    new_params = torch.split(torch.tensor(parameters), [p.numel() for p in opt_network.parameters()])
    with torch.no_grad():
        for param, new_param_value in zip(opt_network.parameters(), new_params):
            param.data.copy_(new_param_value.reshape(param.data.shape))
    predicted_y = opt_network(data_x)
    criterion = nn.MSELoss()
    return criterion(data_y, predicted_y).item()

# load the population of GA, GA_sharing and GA_dynamic
# sgd_params = np.load("results/GA/sgd_params.npy")
ga_params = np.load("results/GA/ga_params.npy")
ga_sharing_params = np.load("results/GA/ga_sharing_params.npy")
ga_dynamic_params = np.load("results/GA/ga_dynamic_params.npy")
print(f"GA population shape: {ga_params.shape}")
print(f"GA_sharing population shape: {ga_sharing_params.shape}")
print(f"GA_dynamic population shape: {ga_dynamic_params.shape}")

# identify the peak set of the populations
ga_dps = DPI(population=ga_params, n_niches=3000, niche_radius=5, obj_f=objective_function)
ga_sharing_dps = DPI(population=ga_sharing_params, n_niches=3000, niche_radius=5, obj_f=objective_function)
ga_dynamic_dps = DPI(population=ga_dynamic_params, n_niches=3000, niche_radius=5, obj_f=objective_function)
print(f"GA population peak set size: {len(ga_dps)}")
print(f"GA_sharing population peak set size: {len(ga_sharing_dps)}")
print(f"GA_dynamic population peak set size: {len(ga_dynamic_dps)}")

# update the populations by the peak set
ga_params = np.array(ga_dps)
ga_sharing_params = np.array(ga_sharing_dps)
ga_dynamic_params = np.array(ga_dynamic_dps)
print(f"Filtered GA population shape: {ga_params.shape}")
print(f"Filtered GA_sharing population shape: {ga_sharing_params.shape}")
print(f"Filtered GA_dynamic population shape: {ga_dynamic_params.shape}")
print()

# evaluate the filtered NN parameters
ga_final_losses = np.array([objective_function(x) for x in ga_params])
ga_sharing_final_losses = np.array([objective_function(x) for x in ga_sharing_params])
ga_dynamic_final_losses = np.array([objective_function(x) for x in ga_dynamic_params])

# construct dictionaries of statistics
mean_loss = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
mean_loss["GA"] = np.mean(ga_final_losses)
mean_loss["GA_sharing"] = np.mean(ga_sharing_final_losses)
mean_loss["GA_dynamic"] = np.mean(ga_dynamic_final_losses)

best_loss = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
best_loss["GA"] = np.min(ga_final_losses)
best_loss["GA_sharing"] = np.min(ga_sharing_final_losses)
best_loss["GA_dynamic"] = np.min(ga_dynamic_final_losses)

std_deviation = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
std_deviation["GA"] = np.std(ga_final_losses, ddof=1)
std_deviation["GA_sharing"] = np.std(ga_sharing_final_losses, ddof=1)
std_deviation["GA_dynamic"] = np.std(ga_dynamic_final_losses, ddof=1)

std_error = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
std_error["GA"] = np.std(ga_final_losses, ddof=1) / np.sqrt(len(ga_final_losses))
std_error["GA_sharing"] = np.std(ga_sharing_final_losses, ddof=1) / np.sqrt(len(ga_sharing_final_losses))
std_error["GA_dynamic"] = np.std(ga_dynamic_final_losses, ddof=1) / np.sqrt(len(ga_dynamic_final_losses))

oneQ = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
oneQ["GA"] = np.percentile(ga_final_losses, 25)
oneQ["GA_sharing"] = np.percentile(ga_sharing_final_losses, 25)
oneQ["GA_dynamic"] = np.percentile(ga_dynamic_final_losses, 25)

median = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
median["GA"] = np.percentile(ga_final_losses, 50)
median["GA_sharing"] = np.percentile(ga_sharing_final_losses, 50)
median["GA_dynamic"] = np.percentile(ga_dynamic_final_losses, 50)

threeQ = {"GA": 0, "GA_sharing": 0, "GA_dynamic": 0}
threeQ["GA"] = np.percentile(ga_final_losses, 75)
threeQ["GA_sharing"] = np.percentile(ga_sharing_final_losses, 75)
threeQ["GA_dynamic"] = np.percentile(ga_dynamic_final_losses, 75)

# List of dictionaries
list_of_dicts = [mean_loss, best_loss, std_deviation, std_error, oneQ, median, threeQ]

# Iterate through the list and print each dictionary
for idx, d in enumerate(list_of_dicts, start=1):
    print(f"Dictionary {idx}:")
    for key, value in d.items():
        print(f"  {key}: {value}")
    print()

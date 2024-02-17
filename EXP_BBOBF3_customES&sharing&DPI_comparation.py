import numpy as np
from data import data_generator
from network import hidden2_FNN
from EXP_BBOBF3_customES import customized_ES as ES
from EXP_BBOBF3_customES_sharing import customized_ES as ES_sharing
from EXP_BBOBF3_customES_dynamic import customized_ES as ES_dynamic
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return best_params, epochs, epoch_losses


def writeFile(filename, mylist):
    # Open a file in write mode ('w')
    with open(filename, 'w') as file:
        # Write each list item to the file
        for item in mylist:
            file.write(f"{item}\n")


# if __name__ == "__main__":

# data generation
f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
data_x, data_y = f_3_d_2_generator.generate(data_size=5000)
data_x.to(device=device)
data_y.to(device=device)
# construct the model and define objective function
ea_network = hidden2_FNN(2, 50, 20, 1)
ea_network.to(device=device)
num_parameters = sum(p.numel() for p in ea_network.parameters())
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
budget_iters = 3000
sgd_lr = 0.0001
criterion = nn.MSELoss()

SGD_losses = np.zeros((n_repeatitions, budget_iters))
ES_losses = np.zeros((n_repeatitions, budget_iters))
ES_sharing_losses = np.zeros((n_repeatitions, budget_iters))
ES_dynamic_losses = np.zeros((n_repeatitions, budget_iters))

SGD_params = []
ES_params = []
ES_sharing_params = []
ES_dynamic_params = []
SGD_results = []
ES_results = []
ES_sharing_results = []
ES_dynamic_results = []

for r in range(n_repeatitions):
    sgd_network = hidden2_FNN(2, 50, 20, 1)
    sgd_network.to(device=device)
    best_individual, epochs, epoch_losses = SGDOpt(sgd_network, data_x, data_y, criterion, budget_iters, sgd_lr)
    population_ES, eval_num, losses, iters, best_losses_ES = ES(dim=num_parameters, budget=budget_iters, mu_=15, lambda_=300, obj_fun=objective_function)
    population_ES_sharing, iters, best_losses_ES_sharing = ES_sharing(dim=num_parameters, budget=budget_iters, mu_=50, lambda_=300, obj_fun=objective_function, niche_radius=5)
    population_ES_dynamic, iters, best_losses_ES_dynamic = ES_dynamic(dim=num_parameters, budget=budget_iters, mu_=50, lambda_=300, obj_fun=objective_function, n_niches=15, niche_radius=5)
    SGD_losses[r] = epoch_losses
    ES_losses[r] = best_losses_ES
    ES_sharing_losses[r] = best_losses_ES_sharing
    ES_dynamic_losses[r] = best_losses_ES_dynamic
    SGD_params.append(best_individual)
    ES_params.append(population_ES)
    ES_sharing_params.append(population_ES_sharing)
    ES_dynamic_params.append(population_ES_dynamic)
    SGD_results.append(objective_function(best_individual))
    ES_results.append([objective_function(x) for x in population_ES])
    ES_sharing_results.append([objective_function(x) for x in population_ES_sharing])
    ES_dynamic_results.append([objective_function(x) for x in population_ES_dynamic])
    print(f"Round {r} is over.")

np.save("SGD_params", np.array(SGD_params))
np.save("ES_params", np.array(ES_params))
np.save("ES_sharing_params", np.array(ES_sharing_params))
np.save("ES_dynamic_params", np.array(ES_dynamic_params))
np.save("SGD_results", np.array(SGD_results))
np.save("ES_results", np.array(ES_results))
np.save("ES_sharing_results", np.array(ES_sharing_results))
np.save("ES_dynamic_results", np.array(ES_dynamic_results))

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
best_loss["ES"] = np.min(ES_final_losses)
best_loss["ES_sharing"] = np.min(ES_sharing_final_losses)
best_loss["ES_dynamic"] = np.min(ES_dynamic_final_losses)

# List of dictionaries
list_of_dicts = [mean_loss, std_deviation, std_error, oneQ, median, threeQ, best_loss]

# Iterate through the list and print each dictionary
for idx, d in enumerate(list_of_dicts, start=1):
    print(f"Dictionary {idx}:")
    for key, value in d.items():
        print(f"  {key}: {value}")
    print()

avg_SGD_losses = np.mean(SGD_losses, axis=0)
avg_ES_losses = np.mean(ES_losses, axis=0)
avg_ES_sharing_losses = np.mean(ES_sharing_losses, axis=0)
avg_ES_dynamic_losses = np.mean(ES_dynamic_losses, axis=0)
# plot the learning curve of loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_SGD_losses, label='SGD')
plt.plot(iters, avg_ES_losses, label='ES')
plt.plot(iters, avg_ES_sharing_losses, label='ES_sharing')
plt.plot(iters, avg_ES_dynamic_losses, label='ES_dynamic')
plt.yscale('log')
plt.title('SGD, a Customized ES, with Fitness Sharing or Dynamic Fitness Sharing for NN Optimization')
plt.xlabel('Number of Iterations / Epochs')
plt.ylabel('MSE Loss (log scale)')
plt.legend()
plt.show()


import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
from nevergrad.optimization import optimizerlib
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

    
# generate data
f_3_d_3_generator = data_generator(suite_name="bbob", function=3, dimension=3, instance=1)
data_x, data_y = f_3_d_3_generator.generate(data_size=1000)


# SGD optimization
sgd_network = hidden2_FNN(3, 50, 20, 1)

# settings
num_epochs = 5000
sgd_learning_rate = 0.001
criterion = nn.MSELoss()
sgd_optimizer = optim.SGD(sgd_network.parameters(), lr=sgd_learning_rate)

# start training
epochs = []
epoch_losses = []
for epoch in range(num_epochs):
    sgd_optimizer.zero_grad()
    # forward pass and evaluate
    predicted_y = sgd_network(data_x)
    epoch_loss = criterion(predicted_y, data_y)
    # backward pass and optimization
    epoch_loss.backward()
    sgd_optimizer.step()
    # record values for plotting
    epochs.append(epoch)
    epoch_losses.append(epoch_loss.item())
    # print the current best value at some intervals
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MSE_Loss = {epoch_loss.item()}")

# get the final best parameters and loss
sgd_best_params = [param.data.tolist() for param in sgd_network.parameters()]
def flatten_list(lst):
    result = []
    for el in lst:
        if isinstance(el, list):
            result.extend(flatten_list(el))
        else:
            result.append(el)
    return result
sgd_best_params = np.array(flatten_list(sgd_best_params))
predicted_y = sgd_network(data_x)
sgd_best_loss = criterion(predicted_y, data_y).item()


# EA optimization
ea_network = hidden2_FNN(3, 50, 20, 1)

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
    # return the loss
    return criterion(data_y, predicted_y).item()

# choose an optimizer
num_parameters = sum(p.numel() for p in ea_network.parameters())
ea_optimizer = optimizerlib.OnePlusOne(parametrization=num_parameters, budget=5000)

# start training
iterations = []
objective_values = []
for i in range(ea_optimizer.budget):
    # evolve
    recommendation = ea_optimizer.ask()
    objective_value = objective_function(recommendation.value)
    ea_optimizer.tell(recommendation, objective_value)
    # record values for plotting
    iterations.append(i)
    objective_values.append(objective_value)
    # print the current best value at some intervals
    if i % 100 == 0:
        print(f"Iteration {i}: MSE_Loss = {objective_function(ea_optimizer.provide_recommendation().value)}")

# get the final best parameters and loss
ea_best_params = ea_optimizer.provide_recommendation().value
ea_best_loss = objective_function(ea_optimizer.provide_recommendation().value)


# print final results
print("SGD Best Parameters:", sgd_best_params)
print("SGD Best Loss:", sgd_best_loss)
print("EA Best Parameters:", ea_best_params)
print("EA Best Loss:", ea_best_loss)

# plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_losses, label='SGD')
plt.plot(iterations, objective_values, label='EA')
plt.yscale('log')
plt.title('SGD vs. EA optimization')
plt.xlabel('Iteration/Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.legend()
plt.show()

import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
from nevergrad.optimization import optimizerlib
from nevergrad.parametrization import parameter
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

    
# data generation
f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
data_x, data_y = f_3_d_2_generator.generate(data_size=5000)

# repeated experiments
n_repeatitions = 20
# training settings
num_epochs = 5000
sgd_learning_rate = 0.0001
criterion = nn.MSELoss()
ea_budget = 5000
# record processes
sgd_training_losses = np.zeros((n_repeatitions, num_epochs))
ea_0_training_losses = np.zeros((n_repeatitions, ea_budget))
ea_1_training_losses = np.zeros((n_repeatitions, ea_budget))
ea_2_training_losses = np.zeros((n_repeatitions, ea_budget))
ea_3_training_losses = np.zeros((n_repeatitions, ea_budget))
ea_4_training_losses = np.zeros((n_repeatitions, ea_budget))
ea_5_training_losses = np.zeros((n_repeatitions, ea_budget))
ea_all_training_losses = [ea_0_training_losses, ea_1_training_losses, ea_2_training_losses, ea_3_training_losses, 
                          ea_4_training_losses, ea_5_training_losses]

# start
for round in range(n_repeatitions):

    # create the NN with SGD optimization
    sgd_network = hidden2_FNN(2, 50, 20, 1)
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
        # if (epoch+1) % 100 == 0:
        #     print(f"Epoch {epoch+1}: MSE_Loss = {epoch_loss.item()}")
    sgd_training_losses[round] = np.array(epoch_losses)
    print("SGD is finished.")

    # create the NN with EA optimization
    ea_network = hidden2_FNN(2, 50, 20, 1)
    num_parameters = sum(p.numel() for p in ea_network.parameters())
    ea_optimizer_0 = optimizerlib.NGOpt(parametrization=num_parameters, budget=ea_budget)
    ea_optimizer_1 = optimizerlib.TwoPointsDE(parametrization=num_parameters, budget=ea_budget)
    ea_optimizer_2 = optimizerlib.OnePlusOne(parametrization=num_parameters, budget=ea_budget)
    ea_optimizer_3 = optimizerlib.CMA(parametrization=num_parameters, budget=ea_budget)
    ea_optimizer_4 = optimizerlib.TBPSA(parametrization=num_parameters, budget=ea_budget)
    ea_optimizer_5 = optimizerlib.RandomSearch(parametrization=num_parameters, budget=ea_budget)
    ea_optimizers = [ea_optimizer_0, ea_optimizer_1, ea_optimizer_2, ea_optimizer_3, ea_optimizer_4, ea_optimizer_5]
    
    ea_number = 0
    for ea_optimizer, ea_training_losses in zip(ea_optimizers, ea_all_training_losses):
        
        # recreate the ea_network for every ea_optimizer
        ea_network = hidden2_FNN(2, 50, 20, 1)

        # define the objective function
        def objective_function(parameters):
            # assign the network with specific parameters
            new_params = torch.split(torch.tensor(parameters), [p.numel() for p in ea_network.parameters()])
            with torch.no_grad():
                for param, new_param_value in zip(ea_network.parameters(), new_params):
                    param.data.copy_(new_param_value.reshape(param.data.shape))
            # evaluate the network with the parameters
            predicted_y = ea_network(data_x)
            # return the loss to be minimized
            return criterion(data_y, predicted_y).item()

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
            # if (i+1) % 100 == 0:
            #     print(f"Iteration {i+1}: MSE_Loss = {objective_function(ea_optimizer.provide_recommendation().value)}")
        
        ea_training_losses[round] = np.array(objective_values)
        print(f"EA {ea_number} is finished.")
        ea_number += 1

    print(f"Round {round+1} is finished.")

avg_sgd_training_losses = np.mean(sgd_training_losses, axis=0)
avg_ea_0_training_losses = np.mean(ea_0_training_losses, axis=0)
avg_ea_1_training_losses = np.mean(ea_1_training_losses, axis=0)
avg_ea_2_training_losses = np.mean(ea_2_training_losses, axis=0)
avg_ea_3_training_losses = np.mean(ea_3_training_losses, axis=0)
avg_ea_4_training_losses = np.mean(ea_4_training_losses, axis=0)
avg_ea_5_training_losses = np.mean(ea_5_training_losses, axis=0)
epochs = np.arange(num_epochs)
iterations = np.arange(ea_budget)
# plot the learning curve of loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_sgd_training_losses, label='SGD')
plt.plot(iterations, avg_ea_0_training_losses, label='NGOpt')
plt.plot(iterations, avg_ea_1_training_losses, label='TwoPointsDE')
plt.plot(iterations, avg_ea_2_training_losses, label='OnePlusOne')
plt.plot(iterations, avg_ea_3_training_losses, label='CMA')
plt.plot(iterations, avg_ea_4_training_losses, label='TBPSA')
plt.plot(iterations, avg_ea_5_training_losses, label='RandomSearch')
plt.yscale('log')
plt.title('SGD vs. EA optimization')
plt.xlabel('Iteration/Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.legend()
plt.show()


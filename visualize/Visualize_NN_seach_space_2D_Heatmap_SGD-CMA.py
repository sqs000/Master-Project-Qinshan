import sys
import os
# Get the current working directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to the sys.path
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import NN_2_parameters
from data import data_generator
from utils import heatmap_2d
import torch.optim as optim
import nevergrad as ng
import numpy as np


if __name__ == "__main__":

    # Generate data (sphere model)
    f_1_d_2_generator = data_generator(suite_name="bbob", function=5, dimension=2, instance=1, device=torch.device("cpu"))
    data_x, data_y = f_1_d_2_generator.generate(data_size=200)


    # Define the objective function (MSE between true values and predicted values) built by the generated data and 2-parameter model
    def loss(x, y):
        network = NN_2_parameters()
        new_weight = torch.FloatTensor([[x, y]])
        network.fc1.weight = nn.Parameter(new_weight)
        predicted_y = network(data_x)
        criterion = nn.MSELoss()
        return criterion(data_y, predicted_y).detach().numpy()

    heatmap_2d(x_range=[-20, 20], y_range=[-20, 20], function=loss)


    # sgd optimization
    num_epochs = 300
    sgd_learning_rate = 0.001
    criterion = nn.MSELoss()
    sgd_model = NN_2_parameters()
    sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=sgd_learning_rate)

    for epoch in range(num_epochs):
        sgd_optimizer.zero_grad()
        # Forward pass
        predicted_y = sgd_model(data_x)
        epoch_loss = criterion(predicted_y, data_y)
        # Backward pass and optimization
        epoch_loss.backward()
        sgd_optimizer.step()
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss.item()}')

    print(f'Parameters: {[param.data.tolist()[0] for param in sgd_model.parameters()][0]}, Loss: {epoch_loss.item()}')


    # cma optimization
    cma_model = NN_2_parameters()
    cma_optimizer = ng.optimizers.CMA(parametrization=2, budget=5000)

    def objective_f(x):
        new_weight = torch.FloatTensor(np.array([x]))
        cma_model.fc1.weight = nn.Parameter(new_weight)
        predicted_y = cma_model(data_x)
        criterion = nn.MSELoss()
        return criterion(data_y, predicted_y).detach().numpy()

    recommendation = cma_optimizer.minimize(objective_f)
    print(f'Parameters: {recommendation.value}, Loss: {objective_f(recommendation.value)}')


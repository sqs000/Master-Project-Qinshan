import sys
import os
# Get the current working directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to the sys.path
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import network
from data import data_generator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import init_weights, euclidean_distance


if __name__ == "__main__":

    # data generation settings
    suite_name = 'bbob'
    function = 1
    dimension = 3
    instance = 1
    data_size=2000
    # model settings
    input_size = 3
    hidden_size = 10
    output_size = 1
    criterion = nn.MSELoss()
    initial_weights = [0.1, 3]
    initial_biases = [0.1, 0.1]
    # training settings
    batch_size = 64
    lr = 0.001
    num_epochs = 300


    # generate data
    f_1_d_3_generator = data_generator(suite_name=suite_name, function=function, dimension=dimension, instance=instance)
    x, y = f_1_d_3_generator.generate(data_size=data_size)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # build models
    model1 = network.FNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer1 = optim.SGD(model1.parameters(), lr=lr)
    model1.apply(init_weights(weight=initial_weights[0], bias=initial_biases[0]))

    model2 = network.FNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer2 = optim.SGD(model2.parameters(), lr=lr)
    model2.apply(init_weights(weight=initial_weights[1], bias=initial_biases[1]))


    # start training
    for epoch in range(num_epochs):

        loss1 = 0.0
        loss2 = 0.0

        for batch_inputs, batch_targets in data_loader:

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # Forward pass
            batch_outputs1 = model1(batch_inputs)
            batch_loss1 = criterion(batch_outputs1, batch_targets)

            batch_outputs2 = model2(batch_inputs)
            batch_loss2 = criterion(batch_outputs2, batch_targets)

            # Backward pass and optimization
            batch_loss1.backward()
            batch_loss2.backward()
            optimizer1.step()
            optimizer2.step()
            loss1 += batch_loss1.item()
            loss2 += batch_loss2.item()

        average_loss1 = loss1 / len(data_loader)
        average_loss2 = loss2 / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss1: {average_loss1}, Average Loss2: {average_loss2}')
        
import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import network
from data import data_generator
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    # data generation settings
    suite_name = 'bbob'
    dimension  = 2
    instance   = 1
    data_size  = 5000
    # model settings
    input_size = 2
    hidden_size_1 = 50
    hidden_size_2 = 20
    output_size = 1
    criterion = nn.MSELoss()
    # training settings
    batch_size = 5000
    lr = 0.00001
    num_epochs = 1000
    # plotting
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8))
    axes = axes.flatten()

    for function in range(1, 25, 1):
        # generate data
        generator = data_generator(suite_name=suite_name, function=function, dimension=dimension, instance=instance, device=torch.device("cpu"))
        x, y = generator.generate(data_size=data_size)
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_values = []
        # build NN
        model = network.hidden2_FNN(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, output_size=output_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # start training
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, batch_targets in data_loader:
                optimizer.zero_grad()
                # Forward pass
                batch_outputs = model(batch_inputs)
                batch_loss = criterion(batch_outputs, batch_targets)
                # Backward pass and optimization
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss.item()
            average_loss = loss / len(data_loader)
            loss_values.append(average_loss)
        print(f'function {function} initial loss: {loss_values[0]}')
        print(f'function {function} final loss: {loss_values[-1]}')
        # plotting
        axes[function-1].plot(np.arange(1, num_epochs+1, 1), np.array(loss_values))
        axes[function-1].set_title(f'function {function}')
        # printing
        print(f'finish function {function}')
    
    plt.tight_layout()
    plt.show()

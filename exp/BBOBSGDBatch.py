import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
from algorithm import SGDBatchOpt
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # data generation
    generator = data_generator(suite_name="bbob", function=1, dimension=2, instance=1, device=torch.device("cuda"))
    data_x, data_y = generator.generate(data_size=5000, standardize=False)

    # training settings
    n_repeatitions = 10
    num_epochs = 500
    sgd_learning_rates = [0.000005, 0.00001, 0.00005, 0.0001]
    batch_sizes = [32, 64, 128, 256, 512]
    criterion = nn.MSELoss()
    sets_training_losses = []
    for i in range(len(sgd_learning_rates)*len(batch_sizes)):
        sets_training_losses.append(np.zeros((n_repeatitions, num_epochs)))
    # start training
    for r in range(n_repeatitions):
        for i, lr in enumerate(sgd_learning_rates):
            for j, bs in enumerate(batch_sizes):
                sgd_network = hidden2_FNN(2, 50, 20, 1)
                sgd_network.to(torch.device("cuda"))
                best_params, best_loss, epochs, epoch_losses = SGDBatchOpt(sgd_network, data_x, data_y, criterion, num_epochs, lr, bs)
                sets_training_losses[i*len(batch_sizes)+j][r] = epoch_losses
                print(f"Set {i*len(batch_sizes)+j} (lr:{lr}, bs:{bs}) is over.")
        print(f"Round {r} is over.")
    plt.figure(figsize=(30, 18))
    for set_number, set_training_losses in enumerate(sets_training_losses):
        i = set_number // len(batch_sizes)
        j = set_number % len(batch_sizes)
        plt.plot(epochs, np.mean(set_training_losses, axis=0), label=f"1r{sgd_learning_rates[i]}-bs{batch_sizes[j]}")
        print(f"1r{sgd_learning_rates[i]}-bs{batch_sizes[j]} loss: {np.mean(set_training_losses, axis=0)[-1]}")
    plt.yscale('log')
    plt.title('The NN training loss curve using SGD (varying the learning rate)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()







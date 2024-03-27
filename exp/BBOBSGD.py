import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
from algorithm import SGDOpt
import torch
import torch.nn as nn
from network import hidden2_FNN
from data import data_generator
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # data generation
    f_3_d_2_generator = data_generator(suite_name="bbob", function=3, dimension=2, instance=1)
    data_x, data_y = f_3_d_2_generator.generate(data_size=5000)

    # training settings
    n_repeatitions = 20
    num_epochs = 1000
    sgd_learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]
    criterion = nn.MSELoss()
    set_0_training_losses = np.zeros((n_repeatitions, num_epochs))
    set_1_training_losses = np.zeros((n_repeatitions, num_epochs))
    set_2_training_losses = np.zeros((n_repeatitions, num_epochs))
    set_3_training_losses = np.zeros((n_repeatitions, num_epochs))
    sets_training_losses = [set_0_training_losses, set_1_training_losses, set_2_training_losses, set_3_training_losses]
    # start training
    for r in range(n_repeatitions):
        for i, lr in enumerate(sgd_learning_rates):
            sgd_network = hidden2_FNN(2, 50, 20, 1)
            best_params, best_loss, epochs, epoch_losses = SGDOpt(sgd_network, data_x, data_y, criterion, num_epochs, lr)
            sets_training_losses[i][r] = epoch_losses
            print(f"Set {i} is over.")
        print(f"Round {r} is over.")
    avg_set_0_training_losses = np.mean(set_0_training_losses, axis=0)
    avg_set_1_training_losses = np.mean(set_1_training_losses, axis=0)
    avg_set_2_training_losses = np.mean(set_2_training_losses, axis=0)
    avg_set_3_training_losses = np.mean(set_3_training_losses, axis=0)
    # print(best_params.shape)
    # plot the learning curve of loss
    print(f"set 0: {avg_set_0_training_losses[-1]}")
    print(f"set 1: {avg_set_1_training_losses[-1]}")
    print(f"set 2: {avg_set_2_training_losses[-1]}")
    print(f"set 3: {avg_set_3_training_losses[-1]}")
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_set_0_training_losses, label='1e^-5')
    plt.plot(epochs, avg_set_1_training_losses, label='5e^-5')
    plt.plot(epochs, avg_set_2_training_losses, label='1e^-4')
    plt.plot(epochs, avg_set_3_training_losses, label='5e^-4')
    plt.yscale('log')
    plt.title('The NN training loss curve using SGD (varying the learning rate)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (log scale)')
    plt.legend()
    plt.show()







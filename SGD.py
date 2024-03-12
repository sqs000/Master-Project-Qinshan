import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from network import hidden2_FNN
from data import data_generator
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SGDBatchOpt(network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    dataset = TensorDataset(data_x, data_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(network.parameters(), lr=sgd_lr)
    epochs = []
    epoch_losses = []
    for epoch in range(n_epochs):
        loss = 0.0
        for batch_inputs, batch_targets in data_loader:
            optimizer.zero_grad()
            batch_outputs = network(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        epoch_loss = loss / len(data_loader)
        epochs.append(epoch)
        epoch_losses.append(epoch_loss)
    final_params = [param.data.tolist() for param in network.parameters()]
    def flatten_list(lst):
        result = []
        for el in lst:
            if isinstance(el, list):
                result.extend(flatten_list(el))
            else:
                result.append(el)
        return result
    final_params = np.array(flatten_list(final_params))
    with torch.no_grad():
        predicted_y = network(data_x)
        final_loss = criterion(predicted_y, data_y).item()
    return final_params, final_loss, epochs, epoch_losses


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
    return best_params, epoch_loss.item(), epochs, epoch_losses


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
            sgd_network.to(device)
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







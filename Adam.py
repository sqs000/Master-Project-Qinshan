import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
import numpy as np


def AdamBatchOpt(network, data_x, data_y, criterion, n_epochs, sgd_lr, batch_size):
    dataset = TensorDataset(data_x, data_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=sgd_lr)
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
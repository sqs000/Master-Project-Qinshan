# to be written
import torch
import torch.nn as nn
from network import FNN
from data import getForestCoverTypeDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import nevergrad as ng
import numpy as np


# Read Forest Cover Type dataset
train_dataset, test_dataset, X_train, y_train, X_test, y_test = getForestCoverTypeDataset(file_path="C:/Users/15852/Desktop/ThirdS/MasterProject/covertype/covtype.data")

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # sgd optimization
# num_epochs = 300
# sgd_learning_rate = 0.001
# criterion = nn.CrossEntropyLoss()
# sgd_model = FNN(54, 20, 7)
# sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=sgd_learning_rate)

# for epoch in range(num_epochs):
#     sgd_model.train()
#     epoch_loss = 0.0
#     for inputs, labels in train_loader:
#         sgd_optimizer.zero_grad()
#         outputs = sgd_model(inputs)
#         loss = criterion(outputs, labels)
#         epoch_loss += loss.item()
#         loss.backward()
#         sgd_optimizer.step()
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}')


# print(f'Parameters: {[param.data.tolist()[0] for param in sgd_model.parameters()][0]}, Loss: {epoch_loss.item()}')


# cma optimization
cma_model = FNN(54, 20, 7)
num_parameters = sum(p.numel() for p in cma_model.parameters())
cma_optimizer = ng.optimizers.CMA(parametrization=num_parameters, budget=5000)

def objective_f(x):
    # Split the 1D array into parts corresponding to the model parameters
    new_params = torch.split(torch.tensor(x), [p.numel() for p in cma_model.parameters()])
    with torch.no_grad():
        for param, new_param_value in zip(cma_model.parameters(), new_params):
            param.data.copy_(new_param_value.reshape(param.data.shape))
    predicted_y = cma_model(X_train)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predicted_y, y_train)
    return loss.item()

recommendation = cma_optimizer.minimize(objective_f)
print(f'Parameters: {recommendation.value}, Loss: {objective_f(recommendation.value)}')


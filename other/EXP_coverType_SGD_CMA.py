import sys
import os
# Get the current working directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to the sys.path
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
import torch
import torch.nn as nn
from network import FNN
from data import getForestCoverTypeDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import nevergrad as ng


# Read Forest Cover Type dataset
train_dataset, test_dataset, X_train, y_train, X_test, y_test = getForestCoverTypeDataset(file_path="C:/Users/15852/Desktop/ThirdS/MasterProject/covertype/covtype.data")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# SGD optimization
num_epochs = 30
sgd_learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
sgd_model = FNN(54, 54, 7)
sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=sgd_learning_rate)

print("Start training using sgd")
for epoch in range(num_epochs):
    sgd_model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        sgd_optimizer.zero_grad()
        outputs = sgd_model(inputs)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        sgd_optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}')


# CMA optimization
cma_model = FNN(54, 54, 7)
num_parameters = sum(p.numel() for p in cma_model.parameters())
cma_optimizer = ng.optimizers.CMA(parametrization=num_parameters, budget=5000)

def objective_f_train(x):
    # Split the 1D array into parts corresponding to the model parameters
    new_params = torch.split(torch.tensor(x), [p.numel() for p in cma_model.parameters()])
    with torch.no_grad():
        for param, new_param_value in zip(cma_model.parameters(), new_params):
            param.data.copy_(new_param_value.reshape(param.data.shape))
    predicted_y = cma_model(X_train)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predicted_y, y_train)
    return loss.item()

print("Start training using cma")
recommendation = cma_optimizer.minimize(objective_f_train)
print(f'Parameters: {recommendation.value}, Loss: {objective_f_train(recommendation.value)}')


print("Start the evaluation of two models")
# Test sgd_model
sgd_model.eval()
with torch.no_grad():
    test_outputs = sgd_model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Accuracy of the model optimized by SGD: {accuracy * 100:.2f}%")

# Test cma_model
def objective_f_test(x):
    new_params = torch.split(torch.tensor(x), [p.numel() for p in cma_model.parameters()])
    with torch.no_grad():
        for param, new_param_value in zip(cma_model.parameters(), new_params):
            param.data.copy_(new_param_value.reshape(param.data.shape))
    test_outputs = cma_model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Accuracy of the model optimized by CMA: {accuracy * 100:.2f}%")

objective_f_test(recommendation.value)
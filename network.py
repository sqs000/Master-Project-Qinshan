import torch.nn as nn


class FNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class NN_2_parameters(nn.Module):
    
    def __init__(self):
        super(NN_2_parameters, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
    
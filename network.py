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


class hidden2_FNN(nn.Module):
    
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(hidden2_FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size_2, output_size)  # Output
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    

class hidden4_FNN(nn.Module):
    
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size):
        super(hidden4_FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)  # Third hidden layer
        self.fc4 = nn.Linear(hidden_size_3, hidden_size_4)  # Fourth hidden layer
        self.fc5 = nn.Linear(hidden_size_4, output_size)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x
    

class NN_2_parameters(nn.Module):
    
    def __init__(self):
        super(NN_2_parameters, self).__init__()
        self.fc1 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
    
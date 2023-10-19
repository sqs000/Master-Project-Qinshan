import network
from data import data_generator
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# generate data
f_1_d_3_generator = data_generator(suite_name='bbob', function=1, dimension=3, instance=1)
x, y = f_1_d_3_generator.generate(data_size=1000)
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# build model
model = network.FNN(input_size=3, hidden_size=8, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# start training
num_epochs = 200

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
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}')

# model.named_parameters()

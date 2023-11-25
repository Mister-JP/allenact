import torch
import torch.nn as nn
import torch.optim as optim
import os

class CoordinateMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(CoordinateMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_probe(data_folder, model_save_path, input_size):
    # Load the data
    data_file = os.path.join(data_folder, 'inference_data.pt')
    data = torch.load(data_file)

    # Initialize the probe model
    probe = CoordinateMLP(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):  # number of epochs can be adjusted
        running_loss = 0.0
        for i, data_pair in enumerate(data):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            inputs = data_pair['memory_tensor']
            target_coordinates = data_pair['target_coordinates_ind']
            outputs = probe(inputs)

            # Compute loss
            loss = criterion(outputs, target_coordinates)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            # if i % 10 == 9:    # print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

    # Save the trained model
    torch.save(probe.state_dict(), model_save_path)
    print('Finished Training. Model saved to', model_save_path)

# Parameters
data_folder = 'probe_data/20231125_130432'  # replace with your actual folder path
model_save_path = os.path.join(data_folder, 'probe_model.pth')
input_size = 512  # the input size to the MLP probe, as per your architecture

# Train the probe
train_probe(data_folder, model_save_path, input_size)

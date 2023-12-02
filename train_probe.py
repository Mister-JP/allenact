import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
torch.manual_seed(0)
random.seed(0)

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

def calculate_accuracy(predictions, targets, threshold=0.1):
    l2_distances = torch.norm(predictions - targets, dim=1)
    accuracy = torch.mean((l2_distances < threshold).float()).item()
    return accuracy

def train_probe(data_folder, model_save_path, input_size, batch_size=32, epochs=1000, learning_rate=0.001):
    # Check if data folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder '{data_folder}' not found.")

    # Load the data
    data_file = os.path.join(data_folder, 'inference_data.pt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")
    data = torch.load(data_file)

    # Split the data into training, validation, and testing sets
    random.shuffle(data)
    train_size = int(0.7 * len(data))
    valid_size = int(0.15 * len(data))
    train_data, valid_data, test_data = data[:train_size], data[train_size:train_size+valid_size], data[train_size+valid_size:]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Initialize the probe model
    probe = CoordinateMLP(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # Adjusts learning rate

    best_valid_loss = float('inf')
    for epoch in range(epochs):
        probe.train()
        running_loss = 0.0
        running_acc = 0.0

        for data_pair in train_loader:
            optimizer.zero_grad()
            inputs = data_pair['memory_tensor']
            target_coordinates = data_pair['target_coordinates_ind']
            outputs = probe(inputs)
            loss = criterion(outputs, target_coordinates)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, target_coordinates)

        # Validation
        probe.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_pair in valid_loader:
                inputs = data_pair['memory_tensor']
                target_coordinates = data_pair['target_coordinates_ind']
                outputs = probe(inputs)
                valid_loss += criterion(outputs, target_coordinates).item()
                valid_acc += calculate_accuracy(outputs, target_coordinates)

        # Update learning rate
        scheduler.step()

        # Early stopping based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(probe.state_dict(), model_save_path)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {running_acc/len(train_loader):.4f}, Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Acc: {valid_acc/len(valid_loader):.4f}')

    # Load the best model
    probe.load_state_dict(torch.load(model_save_path))

    # Test the model
    probe.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data_pair in test_loader:
            inputs = data_pair['memory_tensor']
            target_coordinates = data_pair['target_coordinates_ind']
            outputs = probe(inputs)
            test_loss += criterion(outputs, target_coordinates).item()
            test_acc += calculate_accuracy(outputs, target_coordinates)

    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc/len(test_loader):.4f}')

# Argument parsing
parser = argparse.ArgumentParser(description='Train CoordinateMLP Probe')
parser.add_argument('--version', type=str, required=True, help='Version string for the run')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
args = parser.parse_args()

# Parameters
VERSION = args.version
data_folder = f'probe_data/{VERSION}'  # replace with your actual folder path
model_save_path = os.path.join(data_folder, 'probe_model.pth')
input_size = 512  # the input size to the MLP probe, as per your architecture

# Train the probe
train_probe(data_folder, model_save_path, input_size, args.batch_size, args.epochs, args.learning_rate)

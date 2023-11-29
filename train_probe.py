import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import random

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
    # Compute the L2 distance between predictions and targets
    l2_distances = torch.norm(predictions - targets, dim=1)
    # Compute accuracy as the percentage of distances below the threshold
    accuracy = torch.mean((l2_distances < threshold).float()).item()
    return accuracy

def train_probe(data_folder, model_save_path, input_size):
    # Load the data
    data_file = os.path.join(data_folder, 'inference_data.pt')
    data = torch.load(data_file)

    # Split the data into training and testing sets (80-20 split)
    random.shuffle(data)
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]


    # Initialize the probe model
    probe = CoordinateMLP(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=0.001)

    loss_history = []
    accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Training loop
    for epoch in range(100):  # number of epochs can be adjusted
        running_loss = 0.0
        for i, data_pair in enumerate(train_data):
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
            # Compute accuracy
            accuracy = calculate_accuracy(outputs, target_coordinates)
            accuracy_history.append(accuracy)

            # Print statistics
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}, accuracy: {accuracy:.3f}')
            running_loss += loss.item()

        loss_history.append(running_loss / len(data))
        # Evaluate on the testing set
        probe.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            running_test_loss = 0.0
            total_test_accuracy = 0.0
            for i, data_pair in enumerate(test_data):
                inputs = data_pair['memory_tensor']
                target_coordinates = data_pair['target_coordinates_ind']
                outputs = probe(inputs)
                test_loss = criterion(outputs, target_coordinates)
                running_test_loss += test_loss.item()
                test_accuracy = calculate_accuracy(outputs, target_coordinates)
                total_test_accuracy += test_accuracy

            avg_test_loss = running_test_loss / len(test_data)
            avg_test_accuracy = total_test_accuracy / len(test_data)
            test_loss_history.append(avg_test_loss)
            test_accuracy_history.append(avg_test_accuracy)

        # Print training and testing statistics
        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.3f}, Train Accuracy: {accuracy:.3f}, Test Loss: {avg_test_loss:.3f}, Test Accuracy: {avg_test_accuracy:.3f}')
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() / 10:.3f}')
    # Plotting and saving the graphs
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the figures
    loss_plot_path = os.path.join(data_folder, f'{VERSION}_loss.png')
    accuracy_plot_path = os.path.join(data_folder, f'{VERSION}_accuracy.png')
    plt.savefig(loss_plot_path)
    plt.savefig(accuracy_plot_path)

    print('Loss and accuracy plots saved.')
            
    # Save the trained model
    torch.save(probe.state_dict(), model_save_path)
    print('Finished Training. Model saved to', model_save_path)

# Argument parsing
parser = argparse.ArgumentParser(description='Train CoordinateMLP Probe')
parser.add_argument('VERSION', type=str, help='Version string for the run')
args = parser.parse_args()

# Parameters
VERSION = args.VERSION
data_folder = f'probe_data/{VERSION}'  # replace with your actual folder path
model_save_path = os.path.join(data_folder, 'probe_model.pth')
input_size = 512  # the input size to the MLP probe, as per your architecture

# Train the probe
train_probe(data_folder, model_save_path, input_size)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data_path = "./fft_all_channels.csv"
data = pd.read_csv(data_path)

# Split data into features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values  # Last column
# Convert labels from 1 and 2 to 0 and 1
y = np.where(y == 1, 0, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
features_train = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
features_test = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)    # Add channel dimension
labels_train = torch.tensor(y_train, dtype=torch.float32)
labels_test = torch.tensor(y_test, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(features_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Define the CNN model
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         # Update the input size of fc1 to match the output size of the conv2 layer after pooling
#         self.fc1 = nn.Linear(32 * 6, 64)  # Adjusted this to 32 * 6
#         self.fc2 = nn.Linear(64, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
#         x = torch.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x

import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)  # Batch normalization for the first conv layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(64)  # Batch normalization for the second conv layer

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 6, 128)  # Increase the number of neurons
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # Apply first conv layer, batch norm, relu, pooling, and dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with relu, dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout layer to prevent overfitting
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification output
        return x


# Train the model with a learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Reduce learning rate if the loss plateaus
    scheduler.step(running_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total += targets.size(0)
        correct += (predicted.view(-1) == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on Test Set: {accuracy:.2f}%')
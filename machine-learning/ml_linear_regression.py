import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
data_path = "./fft_all_channels_states.csv"
data = pd.read_csv(data_path)

# Split data into features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values  # Last column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Convert data to PyTorch tensors

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


features_train = torch.tensor(X_train, dtype=torch.float32)
labels_train = torch.tensor(y_train, dtype=torch.float32)
features_test = torch.tensor(X_test, dtype=torch.float32)
labels_test = torch.tensor(y_test, dtype=torch.float32)

# features_train = torch.tensor(X_train_scaled, dtype=torch.float32)
# features_test = torch.tensor(X_test_scaled, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(features_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(32, 1)  # 32 input features to 1 output


    def forward(self, x):
        return self.linear(x)


# Initialize model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
total_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        predictions = model(inputs)
        loss = criterion(predictions, targets.view(-1, 1))
        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)
print(f'Average Loss on Test Set: {avg_loss:.4f}')
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
# Convert labels from 1 and 2 to 0 and 1
y = np.where(y == 1, 0, 1)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check X_train 's state = 1 and =2's percentage
print("y_train's state = 1's percentage: ", len(y_train[y_train == 0])/len(y_train))
print("y_test's state = 1's percentage: ", len(y_test[y_test == 0])/len(y_test))
import torch
import torch.nn as nn
from scipy.io import loadmat
import torch.nn.functional as F

# Define your model architecture (same as before)
# Define your model architecture (include all layers)
class my_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, (44, 4))
        self.bn1 = nn.BatchNorm2d(20, affine=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.bn3 = nn.BatchNorm2d(1, affine=False)
        self.pool1 = nn.MaxPool2d((1, 10))
        self.conv2 = nn.Conv2d(20, 10, (1, 1))
        self.conv3 = nn.Conv2d(5, 1, (5, 10))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.pool3 = nn.MaxPool2d((1, 1))
        self.fc1 = nn.Linear(70, 32)
        self.fc2 = nn.Linear(32, 2)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Uncomment if used during training
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 128
temp = loadmat('../cwt-python/EEG_mental_states.mat')
data = temp['cwt_data']
label = temp['labels']
del temp

data = torch.tensor(data, device=device, dtype=torch.float)
data = torch.permute(data, (3, 0, 1, 2))
label = torch.tensor(label, device=device, dtype=torch.long)

temp = torch.randperm(900)
data = data[torch.cat((temp[:480], torch.arange(900, 1380))), :, :, :]
label = label[torch.cat((temp[:480], torch.arange(900, 1380)))]

label = torch.tensor(label, device=device, dtype=torch.long).squeeze()

temp = torch.randperm(960)
data_train = data[temp[:600], :, :, :]
data_test = data[temp[600:], :, :, :]
label_train = label[temp[:600]]
label_test = label[temp[600:]]


#  just pick random 60 samples from the data
my_new_data = data[temp[:60], :, :, :]
my_new_label = label[temp[:60]]

print(my_new_label)




# Create a new instance of your model
model = my_cnn()
model.to(device)

# Load the saved state dictionary
model.load_state_dict(torch.load('my_cnn_model.pth', map_location=device))
model.eval()
print("Model loaded from my_cnn_model.pth")

# Now, you can use the model for prediction
# For example, predict on test data
'''
with torch.no_grad():
    # Assuming 'data_test' and 'label_test' are your test datasets
    # Move data to the same device as the model
    data_test = data_test.to(device)
    label_test = label_test.to(device)

    logits = model(data_test)
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == label_test).sum().item()
    total = label_test.size(0)
    accuracy = correct * 100.0 / total
    print(f"Test Accuracy: {accuracy:.2f}%")
'''




# Example: Predicting on new data
# new_data should be a tensor of shape [batch_size, 10, height, width]
# new_data = torch.tensor(my_new_data, device=device, dtype=torch.float)

# Ensure new_data and my_new_label are on the correct device
new_data = my_new_data.to(device)
my_new_label = my_new_label.to(device)

print(new_data.shape)

# Preprocess your new data as needed
# new_data = preprocess(new_data)

# Optional: Define class labels if you have them
class_labels = {0: 'Resting', 1: 'Active'}


with torch.no_grad():

    logits = model(new_data)
    probabilities = F.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    probabilities_percentage = probabilities * 100



    for i in range(new_data.size(0)):
        real_label_idx = my_new_label[i]
        real_label = class_labels.get(real_label_idx, f"Unknown({real_label_idx})")
        pred_class_idx = predicted_classes[i]
        pred_class_label = class_labels.get(pred_class_idx, f"Unknown({pred_class_idx})")
        print(f"Sample {i+1}:")
        print(f"Real Label: {real_label}")
        print(f"Predicted Class: {pred_class_label}")
        print(f"Resting Probability: {probabilities_percentage[i, 0]:.2f}%")
        print(f"Active Probability: {probabilities_percentage[i, 1]:.2f}%")
        print("-------------------------")
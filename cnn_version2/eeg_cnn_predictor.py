import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne

# Define your model architecture (same as used during training)
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

class EEGCNNPredictor:
    def __init__(self, model_path, device=None):
        """
        Initializes the EEG CNN Predictor.

        Parameters:
        - model_path (str): Path to the saved PyTorch model (.pth file).
        - device (torch.device, optional): Device to run the model on. Defaults to CPU or GPU if available.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize and load the model
        self.model = my_cnn()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

        # Define class labels
        self.class_labels = {0: 'Resting', 1: 'Active'}

    def preprocess(self, power_data, n_channels=10):
        """
        Preprocesses the CWT power data for the CNN.

        Parameters:
        - power_data (numpy.ndarray): CWT power data with shape (1, n_channels, n_frequencies, n_times).
        - n_channels (int): Number of channels to select for the CNN input.

        Returns:
        - torch.Tensor: Preprocessed tensor ready for CNN input.
        """
        power_tensor = torch.tensor(power_data, dtype=torch.float32)

        # Select channels
        actual_channels = power_tensor.shape[1]
        if actual_channels >= n_channels:
            selected_channels = list(range(n_channels))
        else:
            # Duplicate channels to reach n_channels
            repeats = n_channels // actual_channels + 1
            selected_channels = (list(range(actual_channels)) * repeats)[:n_channels]

        power_selected = power_tensor[:, selected_channels, :, :]
        power_selected = power_selected.to(self.device)
        return power_selected

    def predict(self, power_data):
        """
        Makes a prediction on the input power data.

        Parameters:
        - power_data (numpy.ndarray): CWT power data with shape (1, n_channels, n_frequencies, n_times).

        Returns:
        - dict: Dictionary containing predicted class and probabilities.
        """
        # Preprocess the data
        input_tensor = self.preprocess(power_data)

        # Perform prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            probabilities_percentage = probabilities * 100

            pred_class_idx = predicted_classes[0].item()
            pred_class_label = self.class_labels.get(pred_class_idx, f"Unknown({pred_class_idx})")
            resting_prob = probabilities_percentage[0, 0].item()
            active_prob = probabilities_percentage[0, 1].item()

        # Return results
        result = {
            'predicted_class_index': pred_class_idx,
            'predicted_class_label': pred_class_label,
            'probabilities': {
                'Resting': resting_prob,
                'Active': active_prob
            }
        }
        return result


# # Initialize the predictor
# model_path = 'my_cnn_model.pth'  # Path to your saved model
# predictor = EEGCNNPredictor(model_path)
#
# # Assume you have collected EEG data and performed CWT
# # For demonstration, let's create dummy power data
# # Replace this with your actual CWT power data
# # power_data should have shape (1, n_channels, n_frequencies, n_times)
# # Example dimensions: (1, 10, 44, 128) - adjust according to your data
#
# # Collect and preprocess your EEG data as before
# # For example:
# # power_data = ... (result from mne.time_frequency.tfr_array_morlet)
#
# # Replace the following line with your actual power_data
# power_data = np.random.rand(1, 10, 44, 150)
#
# # Make a prediction
# result = predictor.predict(power_data)
#
# # Print the results
# print(f"Predicted Class: {result['predicted_class_label']}")
# print(f"Probabilities:")
# for class_label, prob in result['probabilities'].items():
#     print(f"  {class_label}: {prob:.2f}%")
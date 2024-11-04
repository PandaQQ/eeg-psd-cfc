import numpy as np
import scipy.io as sio
from eeg_cnn_predictor import EEGCNNPredictor  # Import EEGCNNPredictor from your custom module

# Load data from the .mat file
file_path = '/Users/pandaqq/EEG-AI/cnn_version2/EEG_mental_states_python.mat'
data = sio.loadmat(file_path)

# Extract CWT data and labels
cwt_data = data['cwt_data']  # Shape: (10, 44, 150, 3076)
labels = data['labels'].flatten()  # Shape: (3076,)

# Define the mapping for class labels
class_mapping = {"Active": 1, "Resting": 0}

# Path to your trained CNN model
model_path = 'my_cnn_model_2.pth'
predictor = EEGCNNPredictor(model_path)  # Initialize predictor with the trained model

# Placeholder for predictions
predictions = []

# Loop over the 3076 samples
for t in range(cwt_data.shape[3]):  # Loop over each time point/sample
    # Extract the (10, 44, 150) data for the current sample
    time_slice = cwt_data[:, :, :, t]  # Shape: (10, 44, 150)

    # Reshape for the predictor if needed (e.g., add batch dimension)
    time_slice_tensor = np.expand_dims(time_slice, axis=0)  # Shape: (1, 10, 44, 150)
    print(time_slice_tensor[0])
    # Perform prediction
    result = predictor.predict(time_slice_tensor)
    predicted_class_label = result['predicted_class_label']

    # Map the predicted class label to 0 or 1
    predicted_class = class_mapping.get(predicted_class_label, -1)  # -1 if label not found
    predictions.append(predicted_class)

    # Print prediction and actual label
    print(f"Sample {t + 1} - Predicted: {predicted_class} ({predicted_class_label}), Actual: {labels[t]}")

# Calculate and print accuracy
accuracy = np.mean(np.array(predictions) == labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
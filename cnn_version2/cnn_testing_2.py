import numpy as np
import mne
import matplotlib.pyplot as plt
from eeg_cnn_predictor import EEGCNNPredictor  # Import EEGCNNPredictor from your custom module

# Load EEG data
file_path = 'ouyang_1008_90Hz.set'
raw = mne.io.read_raw_eeglab(file_path, preload=True)
raw.resample(150)  # Resample to 150 Hz

# Define the list of desired channels
desired_channels = ['TP9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'TP10', 'O1', 'Oz', 'O2']

# ['TP9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'TP10', 'O1', 'Oz', 'O2']

# Check if all desired channels are present and pick only those channels
available_channels = set(raw.ch_names)
missing_channels = set(desired_channels) - available_channels
if missing_channels:
    print(f"Warning: The following desired channels are missing from the data: {missing_channels}")

# Pick the desired channels
raw.pick(desired_channels)

# Convert from microvolts to volts
eeg_data_in_volts = raw.get_data() * 1e6
raw._data = eeg_data_in_volts  # Update MNE object data

# Parameters
model_path = 'my_cnn_model_2.pth'  # Path to the trained CNN model
predictor = EEGCNNPredictor(model_path)  # Instantiate the predictor
sampling_rate = int(raw.info['sfreq'])
frequencies = np.linspace(1, 60, 44)  # Frequency range for CWT
n_cycles = frequencies / 2.0
chunk_duration = 1.0
chunk_samples = int(sampling_rate * chunk_duration)

# Process each second of EEG data
predictions = []
predictions_labels = []
for sec in range(int(raw.times[-1])):
    # Extract data for the current second
    start, stop = sec * sampling_rate, (sec + 1) * sampling_rate
    data, _ = raw[:, start:stop]  # Shape: (n_channels, n_times)

    # Initialize a list to store power values for each channel
    channel_powers = []

    # Perform CWT for each channel independently
    for ch in range(data.shape[0]):
        # Extract the data for the current channel
        channel_data = data[ch, :].reshape(1, 1, -1)  # Reshape to (1, n_channels, n_times) for MNE function

        # Compute CWT for the current channel
        power = mne.time_frequency.tfr_array_morlet(
            channel_data,
            sfreq=sampling_rate,
            freqs=frequencies,
            n_cycles=n_cycles,
            output='complex',
            zero_mean=True
        )  # Power shape: (1, 1, n_frequencies, n_times)

        # Remove single-dimensional entries and store power data
        channel_powers.append(power.squeeze())  # Shape after squeeze: (n_frequencies, n_times)

    # Stack channel power data to form a single array for prediction
    cwt_data = np.stack(channel_powers, axis=0)  # Shape: (n_channels, n_frequencies, n_times)
    cwt_data = np.abs(cwt_data)  # Use absolute values of the complex CWT coefficients
    # Reshape cwt_data for the predictor if needed (e.g., add batch dimension)
    cwt_data_tensor = np.expand_dims(cwt_data, axis=0)  # Shape: (1, n_channels, n_frequencies, n_times)

    # Make a prediction
    result = predictor.predict(cwt_data_tensor)
    predicted_class = result['predicted_class_label']

    # Map class to numeric values (0 for Resting, 2 for Active)
    mapped_class = 2 if predicted_class == "Active" else 1
    predictions.append(mapped_class)


    # Print prediction details
    print(f"Second {sec + 1} Prediction: Class {predicted_class}")
    print("Probabilities:")
    for class_label, prob in result['probabilities'].items():
        print(f"  {class_label}: {prob:.2f}%")

    # Optional: Plot TFR for the first channel of the current second
    # plt.figure(figsize=(10, 6))
    # plt.imshow(
    #     cwt_data[0],  # TFR data for the first channel
    #     aspect='auto',
    #     origin='lower',
    #     extent=[0, chunk_duration, frequencies[0], frequencies[-1]],
    #     interpolation='nearest'
    # )
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title(f'Time-Frequency Representation - Channel {desired_channels[0]}, Second {sec + 1}')
    # plt.colorbar(label='Power')
    # plt.show()

# print("All predictions:", predictions)

# Generate time values for the x-axis
# seconds = np.arange(len(predictions))
#
# # Plot the line for predictions
# plt.figure(figsize=(12, 4))
# plt.plot(seconds, predictions, color='blue', linestyle='-', linewidth=1)
#
# # Overlay points with color based on state (0 for green, 2 for red)
# colors = ['red' if p == 2 else 'green' for p in predictions]
# plt.scatter(seconds, predictions, color=colors, s=20, zorder=3)
#
# # Formatting the plot
# plt.xlabel("Time (seconds)")
# plt.ylabel("State")
# plt.yticks([0, 2], labels=["Resting", "Active"])
# plt.title("EEG Activity Over Time")
# plt.show()
import numpy as np
import mne
import matplotlib.pyplot as plt
from eeg_cnn_predictor import EEGCNNPredictor  # Import EEGCNNPredictor from your custom module

# Load EEG data
file_path = 'ouyang_1008_90Hz.set'
raw = mne.io.read_raw_eeglab(file_path, preload=True)
raw.resample(150)  # Resample to 100 Hz

# Define the list of desired channels
desired_channels = ['TP9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'TP10', 'O1', 'Oz', 'O2']

# Check if all desired channels are present and pick only those channels
available_channels = set(raw.ch_names)
missing_channels = set(desired_channels) - available_channels
if missing_channels:
    print(f"Warning: The following desired channels are missing from the data: {missing_channels}")

# Pick the desired channels
raw.pick(desired_channels)

# Convert from microvolts to volts
eeg_data_in_volts = raw.get_data() * 1e6
# If you want to replace the data in the MNE object itself:
raw._data = eeg_data_in_volts

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
for sec in range(int(raw.times[-1])):
    # Extract data for the current second
    start, stop = sec * sampling_rate, (sec + 1) * sampling_rate
    data, _ = raw[:, start:stop]  # Shape: (n_channels, n_times)

    # Reshape data for tfr_array_morlet (add an epoch dimension)
    data_chunk = data[np.newaxis, :, :]  # Shape becomes (1, n_channels, n_times)

    # Perform time-frequency decomposition using Morlet wavelets
    power = mne.time_frequency.tfr_array_morlet(
        data_chunk,
        sfreq=sampling_rate,
        freqs=frequencies,
        n_cycles=n_cycles,
        output='complex',
        zero_mean=True
    )  # Power shape: (n_epochs, n_channels, n_frequencies, n_times)




    print(f"Power shape: {power.shape}")

    # Make a prediction
    result = predictor.predict(power)
    predicted_class = result['predicted_class_label']
    predictions.append(predicted_class)

    print(f"Second {sec + 1} Prediction: Class {predicted_class}")
    print("Probabilities:")
    for class_label, prob in result['probabilities'].items():
        print(f"  {class_label}: {prob:.2f}%")

    # # Plot TFR for the first channel
    # plt.figure(figsize=(10, 6))
    # plt.imshow(
    #     power[0, 0, :, :],
    #     aspect='auto',
    #     origin='lower',
    #     extent=[0, chunk_duration, frequencies[0], frequencies[-1]],
    #     interpolation='nearest'
    # )
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title(f'Time-Frequency Representation - Channel 1, Second {sec + 1}')
    # plt.colorbar(label='Power')
    # plt.show()

# print("All predictions:", predictions)
import numpy as np
from pylsl import StreamInlet, resolve_stream
import mne
import matplotlib.pyplot as plt

# At the top of your script
from eeg_cnn_predictor import EEGCNNPredictor

# After defining the desired channels and before the while loop
model_path = 'my_cnn_model.pth'
predictor = EEGCNNPredictor(model_path)

# Resolve the EEG stream
print("Looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# Create an inlet to receive data from the stream
inlet = StreamInlet(streams[0])

# Get stream info to extract sampling frequency and channel information
info = inlet.info()
sfreq = info.nominal_srate()
n_channels = info.channel_count()

# Extract channel names from the stream info
ch_names = []
ch = info.desc().child('channels').child('channel')
for _ in range(n_channels):
    ch_names.append(ch.child_value('label'))
    ch = ch.next_sibling()

print("Available channels:", ch_names)

# Define the list of desired channels
desired_channels = ['TP9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'TP10', 'O1', 'Oz', 'O2']

# Find indices of the desired channels
desired_indices = [ch_names.index(ch) for ch in desired_channels if ch in ch_names]

# Check if all desired channels are present
missing_channels = set(desired_channels) - set(ch_names)
if missing_channels:
    print(f"Warning: The following desired channels are missing from the stream: {missing_channels}")
    # Decide whether to proceed or handle the missing channels
    # For now, we'll proceed with the available channels
    # Optionally, you can exit or raise an exception here

# Update n_channels to reflect the number of desired channels
n_channels = len(desired_indices)

# Update ch_names to only include the desired channels
ch_names = [ch_names[i] for i in desired_indices]

# Create an MNE Info object
mne_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Set parameters for time-frequency analysis
frequencies = np.logspace(np.log10(1), np.log10(60), num=44)  # Frequencies from 1 to 60 Hz
n_cycles = frequencies / 2.0  # Number of cycles in Morlet wavelet

# Define chunk duration in seconds and calculate the number of samples per chunk
chunk_duration = 1.0  # seconds
chunk_samples = int(sfreq * chunk_duration)

print("Collecting data and computing CWT...")

while True:
    # Initialize lists to store samples and timestamps
    samples = []
    timestamps = []
    samples_collected = 0

    # Collect data until we have the desired number of samples
    while samples_collected < chunk_samples:
        # Pull a chunk of data from the inlet
        chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=chunk_samples - samples_collected)
        if ts:
            samples.append(chunk)
            timestamps.extend(ts)
            samples_collected += len(chunk)

    # Check if any data was collected
    if not samples:
        continue

    # Concatenate samples and transpose to get shape (n_total_channels, n_times)
    data_chunk_full = np.concatenate(samples, axis=0).T

    # Ensure the data chunk has the correct number of samples
    if data_chunk_full.shape[1] != chunk_samples:
        print(f"Data chunk has {data_chunk_full.shape[1]} samples, expected {chunk_samples}. Skipping this chunk.")
        continue

    # Select only the desired channels
    data_chunk = data_chunk_full[desired_indices, :]  # Shape: (n_desired_channels, n_times)

    # Reshape data for tfr_array_morlet (add an epoch dimension)
    data_chunk = data_chunk[np.newaxis, :, :]  # Shape becomes (1, n_desired_channels, n_times)

    # Perform time-frequency decomposition using Morlet wavelets
    power = mne.time_frequency.tfr_array_morlet(
        data_chunk,
        sfreq=sfreq,
        freqs=frequencies,
        n_cycles=n_cycles,
        output='power',
        zero_mean=True  # Explicitly set to avoid warnings
    )

    print(f"Power shape: {power.shape}")

    # Make a prediction
    result = predictor.predict(power)

    print(result)
    # Print the results
    print(f"Predicted Class: {result['predicted_class_label']}")
    print("Probabilities:")
    for class_label, prob in result['probabilities'].items():
        print(f"  {class_label}: {prob:.2f}%")
    # power has shape (n_epochs, n_channels, n_frequencies, n_times)

    # Process the power data as needed
    # For example, plot the TFR of the first desired channel
    '''
    plt.figure(figsize=(10, 6))
    plt.imshow(
        power[0, 0, :, :],
        aspect='auto',
        origin='lower',
        extent=[0, chunk_duration, frequencies[0], frequencies[-1]],
        interpolation='nearest'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Time-Frequency Representation - Channel {ch_names[0]}')
    plt.colorbar(label='Power')
    plt.show()
    '''
    # Optional: Include a condition to break the loop if needed
    # break
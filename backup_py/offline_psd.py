import mne
import numpy as np
from matplotlib import pyplot as plt

# print mne version
print(mne.__version__)

# Load the EEGLAB .set file
file_path = '../dataset/ouyang_0725.set'
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Extract events and event IDs correctly
events, event_id_map = mne.events_from_annotations(raw)
# print("Event IDs:", event_id_map)
# print("Events array:", events)

# Print the type and shape of the events array to ensure it's correct
print(type(events), events.shape)

# Adjust your event_id dictionary to match the actual events
event_id = {
    'S 33': 5,
}

tmin, tmax = -0.0, 30  # Time before and after the event (in seconds)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
print(epochs)


# Assuming 'epochs' is your Epochs object
# Define your frequency bands
iter_freqs = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 3.75},
    {'name': 'Theta', 'fmin': 3.75, 'fmax': 7.5},
    {'name': 'Alpha', 'fmin': 7.5, 'fmax': 12.5},
    {'name': 'Beta', 'fmin': 12.5, 'fmax': 35},
]
# Assuming 'epochs' is your Epochs object
# Compute the PSD using the Welch method for the entire frequency range
spectrum = epochs.compute_psd(method='welch', fmin=0, fmax=35, n_fft=2048, n_per_seg=1024, picks='eeg')
data, freqs = spectrum.get_data(return_freqs=True)

# Plot the PSD for each defined frequency band
for band in iter_freqs:
    # Find indices of frequencies within the band
    idx_band = np.where((freqs >= band['fmin']) & (freqs <= band['fmax']))[0]

    if idx_band.size > 0:  # Check if indices are found
        # Extract the PSD data for the frequencies within the band
        psd_data = data[:, :, idx_band]

        # Average the PSDs over epochs and channels for the band
        psd_mean = 10 * np.log10(psd_data).mean(axis=(0, 2))  # Convert power to dB, then average

        # Ensure the length of `psd_mean` and `freqs[idx_band]` are the same
        assert len(psd_mean) == len(freqs[idx_band]), "Mismatch in the length of PSD and frequency data."

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(freqs[idx_band], psd_mean, label=f"{band['name']} ({band['fmin']}-{band['fmax']} Hz)")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB)')
        plt.title(f"PSD for {band['name']} Band")
        plt.legend()
        plt.show()
    else:
        print(f"No frequencies found in the {band['name']} band ({band['fmin']}-{band['fmax']} Hz)")
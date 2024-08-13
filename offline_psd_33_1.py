import mne
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load EEGLAB data
raw = mne.io.read_raw_eeglab(r'./dataset/ouyang_0725.set', preload=True)
events, event_id_map = mne.events_from_annotations(raw)
print("Event ID and description:", event_id_map)  # Display event IDs and descriptions

# Step 2: Define your event ID for 'S 33'
event_id = {'S 33': 5}

# Step 3: Find the timestamps for the 'S 33' event
s33_times = events[events[:, 2] == event_id['S 33'], 0]
# drop the last event if it is not a complete event
s33_times = s33_times[:-1]

# Step 4: Calculate PSD
psds_list = []
freqs_list = []

for start in s33_times:
    tmin = start / raw.info['sfreq']  # Start time in seconds
    tmax = tmin + 30  # End time in seconds
    # Select specific time segment of data
    raw_selection = raw.copy().crop(tmin=tmin, tmax=tmax)
    raw_selection.pick_channels(['Cz'])  # Selecting the 'Cz' channel
    data = raw_selection.get_data(picks='eeg')
    psds, freqs =  mne.time_frequency.psd_array_multitaper(data, sfreq=raw.info['sfreq'], fmin=0, fmax=35)
    # psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=raw.info['sfreq'], fmin=0, fmax=35, n_per_seg=None)
    psds_list.append(psds)
    freqs_list.append(freqs)

# Step 5: Average PSD and visualize results
if psds_list:
    avg_psds = np.mean(psds_list, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_list[0], 10 * np.log10(avg_psds.T))  # Convert to dB
    plt.title('Average Power Spectral Density for event S 33 over 0 to 30 seconds')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.show()
else:
    print("No valid 'S 33' event segments found.")
import mne
import numpy as np
import pandas as pd

# Load the EEG data
raw = mne.io.read_raw_eeglab('../dataset_90hz/ouyang_0725_90Hz.set', preload=True)
events, event_id_map = mne.events_from_annotations(raw)

# Extract event timestamps for 'S 22' and 'S 33'
s22_times = events[events[:, 2] == event_id_map['S 22'], 0]
s33_times = events[events[:, 2] == event_id_map['S 33'], 0]

# Ensure that there is an even number of 'S 33' events for processing
if len(s33_times) % 2 != 0:
    s33_times = s33_times[:-1]  # Drop the last event if it's not complete

# Merge and sort the times
lats = np.sort(np.concatenate((s22_times, s33_times)))

# Initialize data and states arrays
all_data = {ch_name: [] for ch_name in raw.ch_names}
states = []

# Loop through the events to extract data segments and states
for j in range(30):
    lat_temp = lats[1 + (j - 1) * 2]
    data_temp = raw.get_data(start=lat_temp + 1, stop=lat_temp + 30 * int(raw.info['sfreq']))

    # Store data for each channel
    for ch_idx, ch_name in enumerate(raw.ch_names):
        all_data[ch_name].extend(data_temp[ch_idx])

    states.extend([1] * (data_temp.shape[1] // int(raw.info['sfreq'])))  # 1 relax

    lat_temp1 = lats[j * 2]
    lat_temp2 = lats[j * 2 + 1]
    data_temp = raw.get_data(start=lat_temp1, stop=lat_temp2)
    data_temp = data_temp[:, :int(raw.info['sfreq']) * (data_temp.shape[1] // int(raw.info['sfreq']))]

    # Store data for each channel
    for ch_idx, ch_name in enumerate(raw.ch_names):
        all_data[ch_name].extend(data_temp[ch_idx])

    states.extend([2] * (data_temp.shape[1] // int(raw.info['sfreq'])))  # 2 task

# Compute the FFT for each segment and calculate the alpha amplitude for all channels
fft_all = {ch_name: [] for ch_name in raw.ch_names}
for ch_name in raw.ch_names:
    data = all_data[ch_name]
    for j in range(len(states)):
        segment = data[j * int(raw.info['sfreq']):(j + 1) * int(raw.info['sfreq'])]
        fft_temp = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), d=1 / raw.info['sfreq'])
        alpha_band = (freqs >= 9) & (freqs <= 13)
        alpha_amplitude = np.mean(np.abs(fft_temp[alpha_band]) * 2 / len(segment))  # Alpha band (9-13 Hz)
        fft_all[ch_name].append(alpha_amplitude)

# Convert fft_all and states to a DataFrame for easier indexing
fft_df = pd.DataFrame(fft_all)
fft_df['states'] = states

# Save the DataFrame to a CSV file
fft_df.to_csv('fft_all_channels_states.csv', index=False)
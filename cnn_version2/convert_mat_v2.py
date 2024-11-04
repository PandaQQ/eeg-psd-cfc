import numpy as np
import mne
from mne.time_frequency import tfr_array_morlet
from scipy.io import savemat

# Load the EEG data
# raw = mne.io.read_raw_eeglab('ouyang_1008_90Hz.set', preload=False)
# raw.resample(150)
#
# import mne

# Load the EEG data
raw = mne.io.read_raw_eeglab('ouyang_1008_90Hz.set', preload=True)
raw.resample(150)


# Display a summary of the dataset
# print("Dataset Summary")
# print("----------------")
# print(f"Sampling Frequency (Hz): {raw.info['sfreq']}")
# print(f"Number of Channels: {raw.info['nchan']}")
# print(f"Channel Names: {raw.ch_names}")
# print(f"Data Duration (seconds): {raw.times[-1]}")
# print(f"Highpass Filter (Hz): {raw.info['highpass']}")
# print(f"Lowpass Filter (Hz): {raw.info['lowpass']}")
#
# # Additional details
# print(f"Data Shape: {raw._data.shape}")  # (n_channels, n_samples)
# print(f"EEG Montage: {raw.get_montage()}")
# print(f"Annotations: {raw.annotations}")
#
# exit()

# show image with size 32 * 1400
# raw.plot(n_channels=32, duration=1400, scalings={'eeg': 100e-6})
# raw.plot_psd()
# exit()
# srate = raw.info['sfreq']
srate = 150

# Convert from microvolts to volts
eeg_data_in_volts = raw.get_data() * 1e6
# If you want to replace the data in the MNE object itself:
raw._data = eeg_data_in_volts

# Get events
events, event_id = mne.events_from_annotations(raw)
event_samples = events[:, 0].astype(int)
event_types = events[:, 2]

# Map event IDs to descriptions
event_descriptions = {v: k for k, v in event_id.items()}
event_types_list = [event_descriptions[e] for e in event_types]

# Get latencies of events 'S 33' and 'S 22'
indices = [i for i, etype in enumerate(event_types_list) if etype in ['S 33', 'S 22']]
lats = event_samples[indices]

# Get channel indices
chs = mne.pick_channels(raw.info['ch_names'], include=['TP9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'TP10', 'O1', 'Oz', 'O2'])

# Prepare frequency parameters for CWT
freqs = np.linspace(1, 60, 44)  # Adjust as needed
n_cycles = freqs / 2.0  # Number of cycles in Morlet wavelet
time_bandwidth = 60  # Set time-bandwidth product

# Initialize lists to collect data
cwt_relax_all_list = []
cwt_calcu_all_list = []

data = raw.get_data()

for ch_j, ch in enumerate(chs):
    print(f'Processing channel {ch_j + 1}/{len(chs)}')
    data_relax = np.array([], dtype=float)
    data_calcu = np.array([], dtype=float)

    for j in range(1, 60):
        # Relax data
        idx_relax = (j - 1) * 2
        lat_temp = int(lats[idx_relax])
        data_temp = data[ch, lat_temp:lat_temp + int(30 * srate)]
        data_relax = np.concatenate((data_relax, data_temp))

        # Calcu data
        lat_temp1 = int(lats[j * 2 - 1])
        lat_temp2 = int(lats[j * 2])
        data_temp = data[ch, lat_temp1:lat_temp2]
        data_temp = data_temp[:int(srate * np.fix(len(data_temp) / srate))]
        data_calcu = np.concatenate((data_calcu, data_temp))

    # Process data_relax
    num_segments_relax = int(np.fix(len(data_relax) / srate))
    cwt_relax_list = []
    for j in range(num_segments_relax):
        temp = data_relax[int(j * srate):int((j + 1) * srate)]
        temp_reshaped = temp[np.newaxis, np.newaxis, :]
        complex_tfr = tfr_array_morlet(temp_reshaped,
                                       srate,
                                       freqs=freqs,
                                       n_cycles=n_cycles,
                                       output='complex',
                                       zero_mean=True)
        wt = complex_tfr[0, 0, :, :]
        cwt_relax_list.append(np.abs(wt))
    cwt_relax_all_ch = np.stack(cwt_relax_list, axis=-1)
    cwt_relax_all_list.append(cwt_relax_all_ch)

    # Process data_calcu
    num_segments_calcu = int(np.fix(len(data_calcu) / srate))
    cwt_calcu_list = []
    for j in range(num_segments_calcu):
        temp = data_calcu[int(j * srate):int((j + 1) * srate)]
        temp_reshaped = temp[np.newaxis, np.newaxis, :]
        complex_tfr = tfr_array_morlet(temp_reshaped,
                                       srate,
                                       freqs=freqs,
                                       n_cycles=n_cycles,
                                       output='complex',
                                       zero_mean=True)
        wt = complex_tfr[0, 0, :, :]
        cwt_calcu_list.append(np.abs(wt))
    cwt_calcu_all_ch = np.stack(cwt_calcu_list, axis=-1)
    cwt_calcu_all_list.append(cwt_calcu_all_ch)

# Stack over channels
cwt_relax_all = np.stack(cwt_relax_all_list, axis=0)
cwt_calcu_all = np.stack(cwt_calcu_all_list, axis=0)

# Concatenate cwt_data
cwt_data = np.concatenate((cwt_relax_all, cwt_calcu_all), axis=-1)

# Create labels
num_relax = cwt_relax_all.shape[-1]
num_calcu = cwt_calcu_all.shape[-1]
labels = np.concatenate((np.zeros(num_relax), np.ones(num_calcu)))

# labels array from (1380,1)
labels = labels[:, np.newaxis]

# Save data
savemat('EEG_mental_states_python.mat', {'cwt_data': cwt_data, 'labels': labels})

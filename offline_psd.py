import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the EEG data
raw = mne.io.read_raw_eeglab('./dataset/ouyang_0725.set', preload=True)
events, event_id_map = mne.events_from_annotations(raw)

# Define event IDs
event_id = {
    'S 22': 4,
    'S 33': 5
}

# Find timestamps for 'S 22' and 'S 33' events
s22_times = events[events[:, 2] == event_id['S 22'], 0]
s33_times = events[events[:, 2] == event_id['S 33'], 0]
s33_times = s33_times[:-1]  # Drop the last event if it's not complete

# Calculate PSDs for 'S 33' events
psds_s33 = []
freqs_s33 = []

for start in s33_times:
    tmin = start / raw.info['sfreq']
    tmax = tmin + 30  # 30 seconds window
    raw_selection = raw.copy().crop(tmin=tmin, tmax=tmax)
    # raw_selection.pick_channels(['Cz', 'Fz', 'Pz', 'Oz'])
    raw_selection.pick_channels(['Cz'])
    # psd, freqs = mne.time_frequency.psd_array_multitaper(raw_selection.get_data(), sfreq=raw.info['sfreq'], fmin=0, fmax=35)
    psd, freqs = mne.time_frequency.psd_array_welch(raw_selection.get_data(), sfreq=raw.info['sfreq'], fmin=0, fmax=35)
    psds_s33.append(psd)

# Calculate PSDs between 'S 22' and 'S 33'
psds_s22_s33 = []

for start in s22_times:
    end = s33_times[s33_times > start]
    if end.size > 0:
        tmin = start / raw.info['sfreq']
        tmax = end[0] / raw.info['sfreq']
        raw_selection = raw.copy().crop(tmin=tmin, tmax=tmax)
        # raw_selection.pick_channels(['Cz', 'Fz', 'Pz', 'Oz'])
        raw_selection.pick_channels(['Cz'])
        # psd, _ = mne.time_frequency.psd_array_multitaper(raw_selection.get_data(), sfreq=raw.info['sfreq'], fmin=0, fmax=35)
        psd, _ = mne.time_frequency.psd_array_welch(raw_selection.get_data(), sfreq=raw.info['sfreq'], fmin=0, fmax=35)
        psds_s22_s33.append(psd)

# Average PSDs and plot
plt.figure(figsize=(12, 6))
if psds_s33:
    avg_psds_s33 = np.mean(psds_s33, axis=0)
    plt.plot(freqs, 10 * np.log10(avg_psds_s33.T), label='Average PSD for S 33')

if psds_s22_s33:
    avg_psds_s22_s33 = np.mean(psds_s22_s33, axis=0)
    plt.plot(freqs, 10 * np.log10(avg_psds_s22_s33.T), label='Average PSD between S 22 and S 33')

plt.title('Comparison of Average Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.legend()
plt.show()
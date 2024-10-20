import mne
from pylsl import StreamInfo, StreamOutlet
import time

# Load EEG data from EEGLAB file
eeglab_file = './ouyang_1008_90Hz.set'
raw = mne.io.read_raw_eeglab(eeglab_file, preload=True)
# resample data to 150 Hz
raw.resample(150)

# Extract data and metadata
data, times = raw.get_data(return_times=True)
n_channels = raw.info['nchan']
sfreq = raw.info['sfreq']
channel_names = raw.info['ch_names']

# Define LSL stream info
info = StreamInfo(name='EEG_Stream',
                  type='EEG',
                  channel_count=n_channels,
                  nominal_srate=sfreq,
                  channel_format='float32',
                  source_id='eeglab_stream')

# Add channel names to stream description
ch_names = info.desc().append_child("channels")
for ch in channel_names:
    ch_names.append_child("channel").append_child_value("label", ch)

# Create outlet to stream data
outlet = StreamOutlet(info)

# Calculate the duration between samples in seconds
sample_interval = 1.0 / sfreq

# Stream data at the same frequency as the original recording
for i in range(data.shape[1]):
    outlet.push_sample(data[:, i])
    temp_data = data[:, i]
    # Pause for the duration of one sample to maintain the original frequency
    time.sleep(sample_interval)
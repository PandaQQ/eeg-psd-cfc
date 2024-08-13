import mne
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read the EEGLAB file
    raw = mne.io.read_raw_eeglab(r'./dataset/ouyang_0725.set')
    # If the event information isn't in the EEGLAB file, you'll need to create events manually or modify this part
    events, event_id_map = mne.events_from_annotations(raw)
    event_id = {
        'S 33': 5,
    }
    # Create epochs from the raw data around event markers
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=30, preload=True, baseline=None)
    # Select a specific channel to analyze
    epochs.pick_channels(['Cz'])
    # Compute the PSD for the epochs
    psd = epochs.compute_psd(fmin=0, fmax=35)
    # Plot the PSD
    psd.plot(average=True, spatial_colors=False)

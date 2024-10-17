from scipy.io import loadmat
import numpy as np

# Load both .mat files
data_matlab = loadmat('../machine-learning/EEG_mental_states.mat')
data_python = loadmat('../cwt-python/EEG_mental_states.mat')


# get the first channel of the data

# Remove metadata
for key in ['__header__', '__version__', '__globals__']:
    data_matlab.pop(key, None)
    data_python.pop(key, None)

# Compare 'cwt_data'
if 'cwt_data' in data_matlab and 'cwt_data' in data_python:
    cwt_data_matlab = data_matlab['cwt_data']
    cwt_data_python = data_python['cwt_data']
    if np.allclose(cwt_data_matlab, cwt_data_python, atol=1e-8):
        print('cwt_data is identical.')
    else:
        print('cwt_data is different.')
else:
    print('cwt_data variable not found in both files.')

# Compare 'labels'
if 'labels' in data_matlab and 'labels' in data_python:
    labels_matlab = data_matlab['labels']
    labels_python = data_python['labels']
    if np.array_equal(labels_matlab, labels_python):
        print('labels are identical.')
    else:
        print('labels are different.')
else:
    print('labels variable not found in both files.')


def main():
    pass
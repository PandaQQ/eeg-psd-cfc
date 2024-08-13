import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import hilbert

def bandpass(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def get_cfc(data_low, data_high, N=18):
    """
    计算相位-振幅耦合的 Modulation Index (MI)。
    参数:
    data_low: 低频段信号的数据
    data_high: 高频段信号的数据
    N: 相位区间的数量，默认是 18
    返回:
    cfc: Modulation Index (MI) 值
    """
    # 计算高频信号的 Hilbert 变换和振幅
    signal_high = hilbert(data_high)
    amp_high = np.abs(signal_high)

    # 计算低频信号的 Hilbert 变换和相位
    signal_low = hilbert(data_low)
    phase_low = np.angle(signal_low)

    # 将相位从弧度转换为角度并将其包装到 0 到 360 度之间
    angle_low = np.rad2deg(phase_low)
    angle_low = np.mod(angle_low, 360)

    # 将相位分段到 N 个区间
    bin_edges = np.linspace(0, 360, N + 1)
    Y = np.digitize(angle_low, bin_edges) - 1  # 分段并得到分段索引

    # 计算每个相位区间的平均振幅
    amp_bin = np.zeros(N)
    for b in range(N):
        ind = np.where(Y == b)[0]
        if len(ind) == 0:
            amp_bin[b] = 0
        else:
            amp_bin[b] = np.mean(amp_high[ind])

    # 对每个相位区间的平均振幅进行归一化
    amp_normal_bin = amp_bin / np.sum(amp_bin)

    # 计算 Shannon 熵
    hp = amp_normal_bin * np.log(amp_normal_bin)
    hp = np.nan_to_num(hp)  # 将 NaN 转换为 0

    # 根据 Kullback-Leibler 距离计算 Modulation Index (MI)
    cfc = (np.log(N) + np.sum(hp)) / np.log(N)

    return np.log(cfc)


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
# Find the channel index for 'Oz'
ch_idx = raw.ch_names.index('Cz')

# Initialize data and states arrays
data = []
states = []

# Loop through the events to extract data segments and states
for j in range(30):
    lat_temp = lats[1 + (j - 1) * 2]
    data_temp = raw.get_data(picks=ch_idx, start=lat_temp + 1, stop=lat_temp + 30 * int(raw.info['sfreq']))
    data.extend(data_temp[0])
    states.extend([1] * (len(data_temp[0]) // int(raw.info['sfreq'])))  # 1 relax

    lat_temp1 = lats[j * 2]
    lat_temp2 = lats[j * 2 + 1]
    data_temp = raw.get_data(picks=ch_idx, start=lat_temp1, stop=lat_temp2)
    data_temp = data_temp[0][:int(raw.info['sfreq']) * (len(data_temp[0]) // int(raw.info['sfreq']))]
    data.extend(data_temp)
    states.extend([2] * (len(data_temp) // int(raw.info['sfreq'])))  # 2 task


# Compute the FFT for each segment and calculate the alpha amplitude
fft_all = []
for j in range(len(states)):
    segment = data[j * int(raw.info['sfreq']):(j + 1) * int(raw.info['sfreq'])]
    fft_temp = np.fft.fft(segment)
    freqs = np.fft.fftfreq(len(segment), d=1/raw.info['sfreq'])
    alpha_band = (freqs >= 9) & (freqs <= 13)
    alpha_amplitude = np.mean(np.abs(fft_temp[alpha_band]) * 2 / len(segment))  # Alpha band (9-13 Hz)
    fft_all.append(alpha_amplitude)


# Convert fft_all to numpy array for easier indexing
fft_all = np.array(fft_all)

# # Find indices for relax and task states
# index_relax = np.where(np.array(states) == 1)[0]
# index_task = np.where(np.array(states) == 2)[0]
#
# # Plot the results
# plt.figure(figsize=(15, 3))
# plt.plot(index_relax, fft_all[index_relax], 'b.')
# plt.plot(index_task, fft_all[index_task], 'r.')
# plt.title('FFT Amplitude')
# plt.xlabel('Index')
# plt.ylabel('Amplitude')
# plt.show()



# Apply bandpass filters
data_highfreq = bandpass(data, 40, 60, int(raw.info['sfreq']))
data_lowfreq = bandpass(data, 6, 7, int(raw.info['sfreq']))

# Compute CFC
cfc = []
for j in range(len(states)):
    data_low = data_lowfreq[j * int(raw.info['sfreq']):(j + 1) * int(raw.info['sfreq'])]
    data_high = data_highfreq[j * int(raw.info['sfreq']):(j + 1) * int(raw.info['sfreq'])]
    cfc.append(get_cfc(data_low, data_high))

# Convert cfc to numpy array for easier indexing
cfc = np.array(cfc)

# Find indices for relax and task states
index_relax = np.where(np.array(states) == 1)[0]
index_task = np.where(np.array(states) == 2)[0]


# Plot the results
plt.figure(figsize=(15, 3))
plt.plot(index_relax, fft_all[index_relax], 'b.')
plt.plot(index_task, fft_all[index_task], 'r.')
plt.title('FFT Amplitude')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.show()


# Plot CFC
plt.figure(figsize=(15, 3))
plt.plot(index_relax, cfc[index_relax], 'b.')
plt.plot(index_task, cfc[index_task], 'r.')
plt.title('CFC')
plt.xlabel('Index')
plt.ylabel('CFC Value')
plt.show()

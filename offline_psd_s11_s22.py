import mne
import numpy as np
import matplotlib.pyplot as plt

# 步骤1: 加载EEGLAB数据
raw = mne.io.read_raw_eeglab(r'./dataset/ouyang_0725.set', preload=True)
events, event_id_map = mne.events_from_annotations(raw)
print("事件ID和描述：", event_id_map)  # 查看事件的ID和描述

# 步骤2: 定义你的事件ID
event_id = {
    'S 22': 4,
    'S 33': 5
}

# 步骤3: 寻找'S 11'和'S 22'事件的时间点
s11_times = events[events[:, 2] == event_id['S 22'], 0]
s22_times = events[events[:, 2] == event_id['S 33'], 0]

# 确保'S 22'事件总是在'S 11'之后，这需要你的数据符合这一逻辑
# 步骤4: 计算PSD
psds_list = []
freqs_list = []

psds_list = []
freqs_list = []

for start in s11_times:
    end = s22_times[s22_times > start]
    print(start, end)

    if end.size > 0:
        tmin = start / raw.info['sfreq']
        tmax = end[0] / raw.info['sfreq']
        # 选择特定时间段的数据
        raw.pick_channels(['Cz'])
        raw_selection = raw.copy().crop(tmin=tmin, tmax=tmax)
        data = raw_selection.get_data(picks='eeg')
        # psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=raw.info['sfreq'], fmin=0, fmax=35)
        psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=raw.info['sfreq'], fmin=0, fmax=35)
        psds_list.append(psds)
        freqs_list.append(freqs)

# 步骤5: 平均PSD和可视化结果
if psds_list:
    avg_psds = np.mean(psds_list, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_list[0], 10 * np.log10(avg_psds.T))  # 转换为分贝
    plt.title('Average Power Spectral Density between S 11 and S 22')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.show()
else:
    print("No valid S 11 and S 33 pairs found.")
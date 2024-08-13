import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_multitaper

mne.set_log_level(False)
iter_freqs = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 3.75},
    {'name': 'Theta', 'fmin': 3.75, 'fmax': 7.5},
    {'name': 'Alpha', 'fmin': 7.5, 'fmax': 12.5},
    {'name': 'Beta', 'fmin': 12.5, 'fmax': 35},
]
# 设置不同事件对应的颜色
# color_events = {'cueLeft': (1, 0, 0), 'cueRight': (0, 1, 0), 'cueFoot': (0, 0, 1), 'cueTongue': (1, 1, 0)}
color_events = {'rest': (1, 0, 0), 'cal': (0, 1, 0)}

#################psd计算频率区间的能量分布(单通道)############
def EpochPSDEnergy(epochs):
    # 遍历epochs中的不同事件
    for event in epochs.event_id:
        # 计算某个事件对应所有epochs的功率谱密度（返回为ndarray）
        psds, freqs = psd_multitaper(epochs[event], n_jobs=1)
        # 计算事件对应epochs的均值
        psds = np.squeeze(np.average(psds, axis=0))
        # 初始化能量矩阵
        eventEnergy = []
        # 遍历不同频率区间的能量和
        for iter_freq in iter_freqs:
            eventEnergy.append(np.sum(psds[(iter_freq['fmin'] < freqs) & (freqs < iter_freq['fmax'])]))
        # 绘制不同事件、不同频率区间的能量值
        plt.plot([xLabel['name'] for xLabel in iter_freqs], eventEnergy, color=color_events[event], label=event,
                 marker='o', lw=0, ms=5)
    # 设置标题
    plt.title('PSD_SUM')
    # 设置图例
    plt.legend()
    # 绘图显示
    plt.show()


if __name__ == '__main__':
    # 读取筛选好的epoch数据
    epochs = mne.read_epochs(r'F:\BaiduNetdiskDownload\BCICompetition\BCICIV_2a_gdf\Train\Fif\A02T_epo.fif')
    # 这里只分析一个通道的psd
    epochs.pick(['EEG-Cz']).plot_psd()
    # 绘制不同区间的能量分布
    EpochPSDEnergy(epochs.pick(['EEG-Cz']))
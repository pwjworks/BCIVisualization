
# 引入python库
import mne
from mne.datasets import sample
import matplotlib.pyplot as plt

# sample的存放地址
data_path = sample.data_path()
# 该fif文件存放地址
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

"""
如果上述给定的地址中存在该文件，则直接加载本地文件，
如果不存在则在网上下载改数据
"""
raw = mne.io.read_raw_fif(fname)


print(raw.info)


"""
案例：
获取10-20秒内的良好的MEG数据

# 根据type来选择 那些良好的MEG信号(良好的MEG信号，通过设置exclude="bads") channel,
结果为 channels所对应的的索引
"""
plt.switch_backend('agg')
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
t_idx = raw.time_as_index([10., 20.])
data, times = raw[picks, t_idx[0]:t_idx[1]]
plt.plot(times, data.T)
plt.title("Sample channels")

plt.savefig("./1.jpg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from scipy import stats

# 数据加载函数
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)  # 假设数据没有表头
    data = data.values[1:127000]  # 跳过第一行
    return data

def fill_na_with_mean(signal):
    series = pd.Series(signal)
    nan_indices = np.where(np.isnan(series))[0]
    for idx in nan_indices:
        prev_value = series.iloc[:idx].dropna().iloc[-1] if idx > 0 else np.nan
        next_value = series.iloc[idx+1:].dropna().iloc[0] if idx < len(series) - 1 else np.nan
        if not np.isnan(prev_value) and not np.isnan(next_value):
            series.iloc[idx] = (prev_value + next_value) / 2
        elif not np.isnan(prev_value):
            series.iloc[idx] = prev_value
        elif not np.isnan(next_value):
            series.iloc[idx] = next_value
    return series.values

# 异常值检测和处理函数
def detect_outliers(signal):
    z_scores = np.abs(stats.zscore(signal))
    return z_scores > 3

def handle_outliers(signal):
    outliers = detect_outliers(signal)
    signal[outliers] = np.nan  # 标记异常值为NaN
    filled_signal = fill_na_with_mean(signal)
    return filled_signal

# 滑动平均滤波函数
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

# 降采样和归一化函数
def downsample(signal, target_length):
    return resample(signal, target_length)

def normalize(signal):
    scaler = MinMaxScaler()
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

# PAA处理函数
def paa(signal, num_segments):
    segment_length = len(signal) // num_segments
    paa_result = np.array([np.mean(signal[i * segment_length: (i + 1) * segment_length]) for i in range(num_segments)])
    return paa_result

# 平滑处理函数
def smooth_transition(signal, threshold=0.01):
    diff = np.diff(signal)
    spikes = np.where(np.abs(diff) > threshold)[0]
    for spike in spikes:
        if spike > 0 and spike < len(signal) - 1:
            signal[spike] = (signal[spike - 1] + signal[spike + 1]) / 2
    return signal

# 读取数据
filepath = './data_demo/Eucalyptus/2024-05-13 10：18：07.txt'
data = load_data(filepath)

# 获取传感器数量
n_sensors = data.shape[1]

# 创建一个图形
plt.figure(figsize=(10, 6))

# 绘制每个传感器的响应曲线在同一张图中
for i in range(n_sensors):
    sensor_data = data[:, i]

    # 处理异常值
    cleaned_data = handle_outliers(sensor_data)

    # 应用滑动平均滤波
    window_size = 3
    smoothed_data = moving_average(cleaned_data, window_size)

    # 降采样到目标长度
    target_length = 640
    downsampled_data = downsample(smoothed_data, target_length)

    # 归一化数据
    normalized_data = normalize(downsampled_data)

    # 使用PAA进一步降维到128
    # target_paa_length = 224
    # paa_data = paa(normalized_data, target_paa_length)

    # 平滑处理
    # smooth_data = smooth_transition(paa_data)

    # 在同一张图上绘制每个传感器的处理后的数据
    plt.plot(normalized_data, label=f'Sensor {i + 1}')

# 添加图例和标签
plt.title('Comparison of Processed Sensor Responses')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Response')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt, decimate
from scipy.integrate import simps
from scipy.fft import fft, fftfreq
import pywt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from scipy import stats


# 数据文件夹路径
# data_folder = './data_demo/Eucalyptus'
# output_folder = './data_demo/Eucalyptus/plus'

data_folder = './wood_gass_data/Eucalyptus'
output_folder = './dataset/Eucalyptus_plus2'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
filter_folder = os.path.join(output_folder, 'Normal')
normal_folder = os.path.join(output_folder, 'Smooth')

# 数据加载函数
def load_data(file_path):
    data = pd.read_csv(file_path)  # 假设数据没有表头
    data = data[9100:127000]
    data = data.values
    return data

def fill_na_with_mean(signal):
    # 将数据转换为 pandas Series
    series = pd.Series(signal)
    
    # 找到 NaN 的索引位置
    nan_indices = np.where(np.isnan(series))[0]
    
    # 对每个 NaN 值进行处理
    for idx in nan_indices:
        # 前一个非 NaN 值
        prev_value = series.iloc[:idx].dropna().iloc[-1] if idx > 0 else np.nan
        # 后一个非 NaN 值
        next_value = series.iloc[idx+1:].dropna().iloc[0] if idx < len(series) - 1 else np.nan
        
        # 如果前后两个值都存在，则取均值
        if not np.isnan(prev_value) and not np.isnan(next_value):
            series.iloc[idx] = (prev_value + next_value) / 2
        # 如果只有前一个值存在，则取前一个值
        elif not np.isnan(prev_value):
            series.iloc[idx] = prev_value
        # 如果只有后一个值存在，则取后一个值
        elif not np.isnan(next_value):
            series.iloc[idx] = next_value
            
    return series.values

# 异常值检测函数
def detect_outliers(signal):
    z_scores = np.abs(stats.zscore(signal))
    return z_scores > 3

# 处理异常值函数
def handle_outliers(signal):
    outliers = detect_outliers(signal)
    signal[outliers] = np.nan  # 标记异常值为NaN
    filled_signal = fill_na_with_mean(signal)
    return filled_signal

# 滤波函数
def gaussian_smoothing(signal, sigma):
    return gaussian_filter(signal, sigma=sigma)

# 降采样函数
def downsample(signal, target_length):
    return resample(signal, target_length)

# 归一化函数
def normalize(signal):
    scaler = MinMaxScaler()
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

def exponential_smoothing(signal, alpha):
    result = np.zeros_like(signal)
    result[0] = signal[0]
    for t in range(1, len(signal)):
        result[t] = alpha * signal[t] + (1 - alpha) * result[t-1]
    return result

# 异常值-滤波-降采样-归一化-平滑
def preprocess_data(data, target_length):
    sensor_data = data

    # 处理异常值
    cleaned_data = handle_outliers(sensor_data)

    # 应用滤波器
    smoothed_gaussian = gaussian_smoothing(cleaned_data, sigma=3)

    # 降采样到目标长度
    downsampled_data = downsample(smoothed_gaussian, target_length)

    # 归一化数据
    normalized_data = normalize(downsampled_data)

    # 对曲线进行平滑
    exponential_signal = exponential_smoothing(normalized_data,alpha=0.01)

    return normalized_data, exponential_signal


def calculate_phase_space_features(signal, embedding_dim=3, time_delay=10):
    # 构建相空间轨迹矩阵
    N = len(signal) - (embedding_dim - 1) * time_delay
    if N <= 0:
        return {}
    
    trajectory_matrix = np.array([signal[i: i + N] for i in range(0, embedding_dim * time_delay, time_delay)]).T
    features = {}

    # 协方差矩阵的特征值
    cov_matrix = np.cov(trajectory_matrix, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    for i, eigenvalue in enumerate(eigenvalues):
        features[f'phase_space_eigenvalue_{i}'] = eigenvalue

    # # 估计最大Lyapunov指数（粗略估计）
    # reg = LinearRegression().fit(np.arange(N).reshape(-1, 1), np.log(np.abs(trajectory_matrix[:, 0])))
    # features['lyapunov_exponent'] = reg.coef_[0]

    return features


def calculate_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_mean_energy_{i}'] = np.mean(np.square(coeff))
        features[f'wavelet_std_{i}'] = np.std(coeff)
        energy_normalized = coeff ** 2 / np.sum(coeff ** 2 + 1e-10)
        features[f'wavelet_entropy_{i}'] = -np.sum(energy_normalized * np.log2(energy_normalized + 1e-10))
    return features

def calculate_frequency_features(signal, sampling_rate):
    N = len(signal)
    T = 1.0 / sampling_rate
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]

    amplitude_spectrum = 2.0 / N * np.abs(yf[0:N//2])
    power_spectrum = np.square(amplitude_spectrum)

    # 计算频率特征
    total_power = np.sum(power_spectrum)
    mean_frequency = np.sum(xf * power_spectrum) / np.sum(power_spectrum)
    bandwidth = np.sqrt(np.sum((xf - mean_frequency) ** 2 * power_spectrum) / np.sum(power_spectrum))

    # 计算谱熵
    power_spectrum_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
    spectral_entropy = -np.sum(power_spectrum_normalized * np.log2(power_spectrum_normalized + 1e-10))

    # 计算谱平坦度
    spectral_flatness = np.exp(np.mean(np.log(power_spectrum + 1e-10))) / (np.mean(power_spectrum) + 1e-10)

    # 计算谱质心
    spectral_centroid = np.sum(xf * power_spectrum) / (np.sum(power_spectrum) + 1e-10)

    return {
        'total_power': total_power,
        'mean_frequency': mean_frequency,
        'bandwidth': bandwidth,
        'spectral_entropy': spectral_entropy,
        'spectral_flatness': spectral_flatness,
        'spectral_centroid': spectral_centroid
    }


# 提取特征
def extract_features(normalized_signal):
    features = {}

    # 统计特征
    peaks, _ = find_peaks(normalized_signal)
    if len(peaks) > 0:
        tp = np.argmax(normalized_signal)
        tmaxd1 = np.argmax(np.diff(normalized_signal, prepend=normalized_signal[0]))
        tmaxd2 = np.argmax(np.diff(np.diff(normalized_signal, prepend=normalized_signal[0]), prepend=0))
        tmind2 = np.argmin(np.diff(np.diff(normalized_signal, prepend=normalized_signal[0]), prepend=0))

        features['f1'] = normalized_signal[tp]
        features['f2'] = np.diff(normalized_signal, prepend=normalized_signal[0])[tmaxd1]
        features['f3'] = np.sqrt(np.mean(normalized_signal**2))
        features['f4'] = np.diff(np.diff(normalized_signal, prepend=normalized_signal[0]), prepend=0)[tmaxd2]
        features['f5'] = np.diff(np.diff(normalized_signal, prepend=normalized_signal[0]), prepend=0)[tmind2]
        features['f6'] = tp
        features['f7'] = tmaxd1
        features['f8'] = tmaxd2
        features['f9'] = tmind2
        features['f10'] = simps(normalized_signal[:tp]) if tp > 0 else 0

    # 频域特征
    sampling_rate = 9.14
    freq_features = calculate_frequency_features(normalized_signal, sampling_rate)
    features.update(freq_features)

    # 小波变换特征
    wavelet_features = calculate_wavelet_features(normalized_signal)
    features.update(wavelet_features)

    # 相空间特征
    phase_space_features = calculate_phase_space_features(normalized_signal)
    features.update(phase_space_features)

    return features

# 处理所有数据文件
all_features = []

# 遍历数据文件夹中的所有文件
for root, _, files in os.walk(data_folder):
    for filename in files:
        if filename.endswith('.txt'):

            # 构造 Filter 和 Normal 数据的保存路径
            sensor_file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(sensor_file_path, data_folder)
            filter_file_path = os.path.join(filter_folder, relative_path)
            normal_file_path = os.path.join(normal_folder, relative_path)

            # 确保目标路径的目录存在
            os.makedirs(os.path.dirname(filter_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(normal_file_path), exist_ok=True)

            # 加载原始数据
            sensor_data = load_data(sensor_file_path)

            # 设置降采样后的数据长度
            target_length = 640

            # 初始化滤波数据和归一化数据数组
            filter_data = np.zeros((target_length, sensor_data.shape[1]))
            normal_data = np.zeros((target_length, sensor_data.shape[1]))

            # 处理并保存 Filter 和 Normal 数据
            for i in range(sensor_data.shape[1]):
                original_signal = sensor_data[:, i]
                filtered_signal, normalized_signal = preprocess_data(original_signal, target_length)

                filter_data[:, i] = filtered_signal
                normal_data[:, i] = normalized_signal

            # 保存处理后的 Filter 和 Normal 数据为TXT格式
            np.savetxt(filter_file_path, filter_data)
            np.savetxt(normal_file_path, normal_data)

            # 提取特征
            file_features = []
            for i in range(sensor_data.shape[1]):
                # original_signal = filter_data[:, i]
                normalized_signal = normal_data[:, i]
                features = extract_features(normalized_signal)
                file_features.append(features)

            all_features.append(file_features)

# 转换为DataFrame
features_df = pd.DataFrame([feat for file_feats in all_features for feat in file_feats])
features_df.to_csv(os.path.join(output_folder, 'Eucalyptus_features7.csv'), index=False)
# print(features_df)
print("Feature extraction completed.")
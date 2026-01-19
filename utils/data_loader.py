"""
MBWCT-Net 数据加载和预处理工具
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from scipy import stats


class GasSensorDataset(Dataset):
    """气体传感器数据集类"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


def load_gas_sensor_data(data_folder):
    """
    从文件夹加载气体传感器数据
    :param data_folder: 包含Eucalyptus, Rosewood, Sandalwood子文件夹的主文件夹路径
    :return: 数据和标签
    """
    all_data = []
    all_labels = []
    label_map = {'Eucalyptus': 0, 'Rosewood': 1, 'Sandalwood': 2}

    for class_name in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_path):
            continue
            
        label = label_map[class_name]
        
        for file_name in os.listdir(class_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(class_path, file_name)
                
                # 加载单个文件的数据
                data = load_single_file(file_path)
                
                # 添加到总数据集中
                all_data.append(data)
                all_labels.append(label)

    # 转换为numpy数组
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    return all_data, all_labels


def load_single_file(file_path):
    """
    加载单个txt文件的数据
    :param file_path: 文件路径
    :return: 处理后的数据
    """
    data = np.loadtxt(file_path)
    
    # 跳过第一行并限制数据范围
    if data.shape[0] >= 127000:
        data = data[1:127000, :]
    else:
        raise ValueError(f"File {file_path} has insufficient data rows")
    
    # 数据预处理步骤
    processed_data = preprocess_data(data)
    
    return processed_data


def preprocess_data(data):
    """
    预处理数据
    :param data: 原始数据
    :return: 预处理后的数据
    """
    # 异常值处理
    cleaned_data = handle_outliers(data)
    
    # 归一化
    normalized_data = normalize_data(cleaned_data)
    
    # 降采样到目标长度
    target_length = 640
    downsampled_data = downsample_data(normalized_data, target_length)
    
    # 转换为正确的形状 (channels, length)
    if downsampled_data.shape[0] != 8:  # 如果是 (length, channels) 需要转置
        downsampled_data = downsampled_data.T
    
    return downsampled_data


def handle_outliers(data):
    """
    处理异常值
    :param data: 输入数据
    :return: 处理后的数据
    """
    # 检测异常值
    z_scores = np.abs(stats.zscore(data, axis=0))
    outliers = z_scores > 3
    
    # 用均值替换异常值
    for i in range(data.shape[1]):  # 对每个通道分别处理
        channel_data = data[:, i]
        channel_outliers = outliers[:, i]
        if np.any(channel_outliers):
            # 计算非异常值的均值
            non_outlier_mean = np.mean(channel_data[~channel_outliers])
            # 替换异常值
            channel_data[channel_outliers] = non_outlier_mean
            data[:, i] = channel_data
    
    return data


def normalize_data(data):
    """
    归一化数据到 [0, 1] 范围
    :param data: 输入数据
    :return: 归一化后的数据
    """
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    return normalized


def downsample_data(data, target_length):
    """
    降采样数据到目标长度
    :param data: 输入数据
    :param target_length: 目标长度
    :return: 降采样后的数据
    """
    downsampled = np.zeros((target_length, data.shape[1]))
    for i in range(data.shape[1]):
        downsampled[:, i] = resample(data[:, i], target_length)
    
    return downsampled


def create_data_loaders(data_path, batch_size=32, test_size=0.2, val_size=0.1):
    """
    创建训练、验证和测试数据加载器
    :param data_path: 数据路径
    :param batch_size: 批次大小
    :param test_size: 测试集比例
    :param val_size: 验证集比例
    :return: 训练、验证和测试数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 加载数据
    data, labels = load_gas_sensor_data(data_path)
    
    # 首先划分训练集和临时集（测试+验证）
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=test_size+val_size, random_state=42, stratify=labels
    )
    
    # 然后将临时集划分为测试集和验证集
    test_size_adjusted = test_size / (test_size + val_size)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=test_size_adjusted, random_state=42, stratify=temp_labels
    )
    
    # 创建数据集
    train_dataset = GasSensorDataset(train_data, train_labels)
    val_dataset = GasSensorDataset(val_data, val_labels)
    test_dataset = GasSensorDataset(test_data, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_processed_data(file_path):
    """
    加载预处理好的数据（如果存在）
    :param file_path: 预处理数据文件路径
    :return: 数据和标签
    """
    with open(file_path, 'rb') as f:
        data = np.load(f)
        labels = np.load(f)
    return data, labels
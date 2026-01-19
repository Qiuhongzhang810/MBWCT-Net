import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
# from util import load_dataset, read_sensor_data_from_file
from resnet1d import ResNet1D, MyDataset
from acnn1d import ACNN, MyDataset
from crnn1d import CRNN, MyDataset
from net1d import Net1D, MyDataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def read_sensor_data_from_file(file_path):
    """
    读取txt文件中的数据，跳过第一行并只读取第2行到第127000行的数据（假设数据格式为127000*8）。
    """
    # data = pd.read_csv(file_path, delimiter=',',skiprows=1).values
    data = np.loadtxt(file_path)

    # if data.shape[0] < 126999 or data.shape[1] != 8:
    #     raise ValueError(f"File {file_path} has unexpected shape {data.shape}")

    # trimmed_data = data[0:126998, :]  # 从第2行到第127000行的数据

    # return trimmed_data
    return data

def load_dataset(data_folder):
    """
    从指定的文件夹加载数据集。数据文件在多层目录中，标签由文件夹名称确定。
    """
    all_X = []
    all_Y = []
    label_map = {'Eucalyptus': 0, 'Rosewood': 1, 'Sandalwood': 2}

    for subfolder_name in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue
        
        if subfolder_name not in label_map:
            print(f"Unknown label folder: {subfolder_name}")
            continue
        
        label = label_map[subfolder_name]
        
        for root, dirs, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        data = read_sensor_data_from_file(file_path)
                        if data.shape == (640, 8):# 126998
                            data = np.transpose(data, (1, 0))
                            all_X.append(data)
                            all_Y.append(label)
                        else:
                            print(f"File {file_path} has unexpected shape {data.shape}")
                    except ValueError as e:
                        print(e)

    all_X = np.array(all_X)
    all_Y = np.array(all_Y)
    return all_X, all_Y

data_folder = 'C:/Users/22209/Desktop/气体论文/Woodgass/PAA640'
# data_folder = '/home/featurize/work/Woodgass/PAA640'

data, label = load_dataset(data_folder)
print(data.shape, Counter(label))

import pywt
import numpy as np

def wavelet_transform(data, wavelet='db4', level=5):
    """
    对输入的多通道信号数据应用多级小波变换。
    参数:
    - data: 一个形状为(n_samples, n_sensors, n_features)的NumPy数组，代表信号数据。
    - wavelet: 使用的小波类型。
    - level: 小波变换的层数。
    
    返回:
    - transformed_data: 小波变换的结果，包括所有层的近似和细节系数。
    """
    transformed_data = []
    for sample in data:
        sample_transformed = []
        for sensor_signal in sample:
            coeffs = pywt.wavedec(sensor_signal, wavelet, level=level)
            # 将所有系数串联起来作为特征
            features = np.concatenate([c.ravel() for c in coeffs])
            sample_transformed.append(features)
        transformed_data.append(sample_transformed)
    return np.array(transformed_data)

# # 示例数据
# # 假设每个样本包含多个传感器，每个传感器的序列长度为1280
# n_samples = 10  # 10个样本
# n_sensors = 8   # 每个样本包含8个传感器
# n_features = 640  # 每个传感器的序列长度
# data = np.random.randn(n_samples, n_sensors, n_features)  # 随机生成数据

# 应用小波变换
transformed_data = wavelet_transform(data)
print("Transformed data shape:", transformed_data.shape)

# 注意：变换后的数据维度将取决于信号的长度和选择的层数

# split data into training and validation sets
data_train, data_val, label_train, label_val = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)

# create datasets and dataloaders
train_dataset = MyDataset(data_train, label_train)
val_dataset = MyDataset(data_val, label_val)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 定义模型
# from torchsummaryX import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_debug = False
model = CRNN(
    in_channels=8, 
    n_len_seg=160, 
    verbose=False,
    n_classes=3,
    device=device)

# summary(model, torch.zeros(1, 1, 3000))

model.to(device)

import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 初始化优化器、损失函数和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, verbose=True)
loss_func = nn.CrossEntropyLoss()
writer = SummaryWriter(logdir='C:/Users/22209/Desktop/气体论文/Woodgass/resnet1d/logs/crnn')
# writer = SummaryWriter(logdir='/home/featurize/work/Woodgass/resnet1d/logs/acnn')


output_dir = 'C:/Users/22209/Desktop/气体论文/Woodgass/resnet1d/output/crnn'
# output_dir = '/home/featurize/work/Woodgass/resnet1d/output/crnn'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
n_epoch = 150
step = 0
prev_f1 = 0
patience_counter = 0  # Early stopping counter
best_val_loss = float('inf')
train_losses, val_losses = [], []

# Training and validation loops
for epoch in tqdm(range(n_epoch), desc="Epoch", leave=False):

    # Training phase
    model.train()
    running_loss = 0.0
    prog_iter = tqdm(train_dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(prog_iter):
        input_x, input_y = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        pred = model(input_x)
        loss = loss_func(pred, input_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        writer.add_scalar('Loss/train', loss.item(), step)

        if is_debug:
            break

    train_loss = running_loss / len(train_dataloader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    writer.add_scalar('Loss/train', loss.item(), step)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    
    # Save model if current validation loss is the lowest
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = os.path.join(output_dir, 'crnn_best.pt')
        torch.save(model.state_dict(), model_path)
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter > 15:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if is_debug:
        break

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.savefig('train_val_loss.png')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model_path, dataloader_test, device, label_names):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_pred_prob = []
    all_true_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader_test, desc="Evaluating", leave=False)):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            all_pred_prob.append(pred.cpu().data.numpy())
            all_true_labels.append(input_y.cpu().data.numpy())

    all_pred_prob = np.concatenate(all_pred_prob)
    all_true_labels = np.concatenate(all_true_labels)
    all_pred = np.argmax(all_pred_prob, axis=1)

    # Calculate and print metrics
    accuracy = accuracy_score(all_true_labels, all_pred)
    precision = precision_score(all_true_labels, all_pred, average='weighted')
    recall = recall_score(all_true_labels, all_pred, average='weighted')
    f1 = f1_score(all_true_labels, all_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(confusion_matrix(all_true_labels, all_pred))
    print(classification_report(all_true_labels, all_pred))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Use the evaluation function with the best model
label_names=['Eucalyptus','Rosewood','Sandalwood']
evaluate_model(model_path, val_dataloader, device, label_names)

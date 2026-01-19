"""
Transformer for 1-d Multi-channel signal data, pytorch

Qiuhong Zhang, Nov 2024

"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


def load_augmented_data(filepath):
    """从指定的文件路径加载增强数据和标签"""
    with open(filepath, 'rb') as f:
        data = np.load(f)
        labels = np.load(f)
    return data, labels

# 调用函数，确保替换以下路径为实际保存增强数据的文件路径
file_path = 'C:/Users/22209/Desktop/气体论文/Woodgass/Augmented640/augmented_data.npy'
data, label = load_augmented_data(file_path)

class SensorDataset(Dataset):
    def __init__(self, data, labels=None, seq_len=640, transform=None, noise_level=0): #0.01
        """
        Args:
            data (numpy array): Shape should be (samples, channels, sequence length).
            labels (numpy array): Corresponding labels of each sample.
            seq_len (int): Length of the input sequence for each training instance.
            transform (callable, optional): Optional transform to be applied on a sample.
            noise_level (float): Standard deviation of Gaussian noise to be added.
        """
        self.data = data
        self.labels = labels
        self.seq_len = seq_len
        self.transform = transform
        self.noise_level = noise_level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :, :self.seq_len]  # Select the first seq_len elements in the sequence

        # Add Gaussian noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, sample.shape)
            sample = sample + noise

        if self.transform:
            sample = self.transform(sample)

        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        return torch.tensor(sample, dtype=torch.float32)

# split data into training and validation sets as rate of 8:2
data_train, data_val, label_train, label_val = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)

# create datasets and dataloaders
train_dataset = SensorDataset(data_train, label_train, seq_len=672)
val_dataset = SensorDataset(data_val, label_val, seq_len=672)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class TransformerModel(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, d_model, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.num_features = num_features  # 特征数量（传感器数）
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 注意这里设置为 True，以适应(batch_size, seq_length, feature_dim)的输入格式
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 从(2700, 672, 8) 转换为 (2700, 672, 8) 适用于batch_first的情况
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # 对序列的输出取平均，以获取全局特征
        x = self.fc_out(x)
        return x

# 模型实例化和参数
num_features = 8  # 传感器数
num_heads = 4
num_layers = 3
d_model = 8  # 特征维度设置为传感器数
dim_feedforward = 256
num_classes = 3
dropout = 0.2

model = TransformerModel(num_features, num_heads, num_layers, d_model, dim_feedforward, num_classes, dropout)

def print_model_summary(model,data):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

print_model_summary(model,data)


def train_model(model, train_loader, val_loader, epochs=150, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)  # 将模型发送到GPU
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_model_path = "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据也发送到GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_val_loss}, Validation Accuracy: {accuracy}%")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    print("Evaluating the best model on the validation set...")
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

# 主函数
if __name__ == '__main__':

    train_model(model, train_dataloader, val_dataloader)

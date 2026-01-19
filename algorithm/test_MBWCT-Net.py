import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from MBWCT_Net import MBWCTNet, MyDataset  # 更新导入路径

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tensorboardX import SummaryWriter
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 导入配置
try:
    from ..config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, EVALUATION_CONFIG
except ImportError:
    # 如果无法导入配置文件，则使用默认值
    MODEL_CONFIG = {
        'in_channels': 8,
        'n_len_seg': 168,
        'n_classes': 3,
        'num_branches': 5,
        'wave': 'db4',
        'J': 2,
        'verbose': False,
        'use_wavelet': True,
        'use_cross_attn': True,
        'use_mlca': False,
        'use_residual': False,
        'use_msat_former': True
    }
    
    TRAINING_CONFIG = {
        'smoothing': 0.038,
        'learning_rate': 0.0005,
        'weight_decay': 0.00006,
        'n_epoch': 150,
        'patience': 15,
        'eta_min': 1e-6,
        'batch_size': 32,
        'test_size': 0.2
    }
    
    PATH_CONFIG = {
        'data_path': 'C:/Users/22209/Desktop/气体论文/Woodgass/Augmented640/augmented_data.npy',
        'output_dir': 'output/mbwct_net',
        'log_dir': 'logs/mbwct_net'
    }
    
    EVALUATION_CONFIG = {
        'label_names': ['Eucalyptus', 'Rosewood', 'Sandalwood']
    }


def load_augmented_data(filepath):
    """从指定的文件路径加载增强数据和标签"""
    with open(filepath, 'rb') as f:
        data = np.load(f)
        labels = np.load(f)
    return data, labels

def prepare_data(data_path, test_size=0.2, batch_size=32):
    """准备训练和验证数据"""
    print(f"Loading data from {data_path}")
    data, label = load_augmented_data(data_path)

    print(f"Original data shape: {data.shape}, Label distribution: {Counter(label)}")

    # 确保数据格式为 (N, C, L) 或 (N, L, C)，模型期望输入为 (B, C, L)
    # 如果数据是 (N, L, C) 格式，需要转置为 (N, C, L)
    if len(data.shape) == 3:
        if data.shape[1] == 8:  # 如果第二个维度是通道数
            # 数据格式为 (N, C, L)，无需调整
            pass
        elif data.shape[2] == 8:  # 如果第三个维度是通道数
            # 数据格式为 (N, L, C)，需要转置为 (N, C, L)
            data = np.transpose(data, (0, 2, 1))
            print(f"Data transposed to shape: {data.shape} (N, C, L)")
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}, expected 3D array")

    print(f"Final data shape: {data.shape}, Expected input format: (batch, 8 channels, length)")

    # split data into training and validation sets
    data_train, data_val, label_train, label_val = train_test_split(
        data, label, test_size=test_size, random_state=42, stratify=label
    )

    # create datasets and dataloaders
    train_dataset = MyDataset(data_train, label_train)
    val_dataset = MyDataset(data_val, label_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def create_model(device, config):
    """根据配置创建模型"""
    model = MBWCTNet(
        in_channels=config.get('in_channels', 8),
        n_len_seg=config.get('n_len_seg', 168),
        n_classes=config.get('n_classes', 3),
        device=device,
        num_branches=config.get('num_branches', 5),
        wave=config.get('wave', 'db4'),
        J=config.get('J', 2),
        verbose=config.get('verbose', False),
        use_wavelet=config.get('use_wavelet', True),
        use_cross_attn=config.get('use_cross_attn', True),
        use_mlca=config.get('use_mlca', False),
        use_residual=config.get('use_residual', False),
        use_msat_former=config.get('use_msat_former', True)
    )
    return model

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        target = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        target = target * (1 - self.smoothing) + (1 - target) * self.smoothing / (input.size(1) - 1)
        return (-target * log_probs).sum(dim=1).mean()

def train_model(model, train_dataloader, val_dataloader, config, device):
    """训练模型的主要函数"""
    # 设置输出目录
    output_dir = config.get('output_dir', 'output/mbwct_net')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化优化器、损失函数和学习率调度器
    loss_func = LabelSmoothingCrossEntropy(smoothing=config.get('smoothing', 0.038))
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.0005), 
        weight_decay=config.get('weight_decay', 0.00006)
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config.get('n_epoch', 150), 
        eta_min=config.get('eta_min', 1e-6), 
        verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.get('log_dir', 'logs/mbwct_net'))
    
    n_epoch = config.get('n_epoch', 150)
    step = 0
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
        writer.add_scalar('Loss/val', val_loss, epoch)
        scheduler.step()  # CosineAnnealingLR不需要传入loss
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save model if current validation loss is the lowest
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, 'mbwct_best.pt')
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            print(f"  -> New best model saved! (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter > config.get('patience', 15):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 关闭TensorBoard writer
    writer.close()
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'train_val_loss.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to: {loss_plot_path}")
    plt.close()
    
    return model_path

def evaluate_model(model, model_path, dataloader_test, device, label_names):
    """评估模型性能"""
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    output_dir = os.path.dirname(model_path)
    conf_matrix = confusion_matrix(all_true_labels, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path)
    print(f"Confusion matrix saved to: {conf_matrix_path}")
    plt.close()

def main():
    """主函数"""
    # 合并所有配置
    full_config = {}
    full_config.update(MODEL_CONFIG)
    full_config.update(TRAINING_CONFIG)
    full_config.update(PATH_CONFIG)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 准备数据
    train_dataloader, val_dataloader = prepare_data(
        full_config['data_path'],
        test_size=full_config['test_size'],
        batch_size=full_config['batch_size']
    )

    # 创建模型
    model = create_model(device, full_config)
    model.to(device)

    # 训练模型
    model_path = train_model(model, train_dataloader, val_dataloader, full_config, device)

    # 评估模型
    print("\n" + "="*50)
    print("Evaluating best model on validation set...")
    print("="*50)
    evaluate_model(model, model_path, val_dataloader, device, EVALUATION_CONFIG['label_names'])

if __name__ == '__main__':
    main()
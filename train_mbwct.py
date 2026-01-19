"""
MBWCT-Net 训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 导入自定义模块
from algorithm.MBWCT_Net import MBWCTNet, MyDataset  # 修正导入语句
from utils.data_loader import create_data_loaders, load_gas_sensor_data
from utils.evaluation import evaluate_model, plot_training_curves
from config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, EVALUATION_CONFIG

# 尝试导入tensorboard，如果没有安装则跳过
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install it with: pip install tensorboard")


def load_data_and_create_loaders():
    """加载数据并创建数据加载器"""
    print("Loading data...")
    
    # 使用 wood_gass_data_demo 文件夹作为示例
    data_path = "./wood_gass_data_demo"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=data_path,
        batch_size=TRAINING_CONFIG['batch_size'],
        test_size=TRAINING_CONFIG['test_size'],
        val_size=0.1
    )
    
    return train_loader, val_loader, test_loader


def create_model_and_optimizer(device):
    """创建模型和优化器"""
    print("Creating model...")
    
    model = MBWCTNet(
        in_channels=MODEL_CONFIG['in_channels'],
        n_len_seg=MODEL_CONFIG['n_len_seg'],
        n_classes=MODEL_CONFIG['n_classes'],
        device=device,
        num_branches=MODEL_CONFIG['num_branches'],
        wave=MODEL_CONFIG['wave'],
        J=MODEL_CONFIG['J'],
        verbose=MODEL_CONFIG['verbose'],
        use_wavelet=MODEL_CONFIG['use_wavelet'],
        use_cross_attn=MODEL_CONFIG['use_cross_attn'],
        use_mlca=MODEL_CONFIG['use_mlca'],
        use_residual=MODEL_CONFIG['use_residual'],
        use_msat_former=MODEL_CONFIG['use_msat_former']
    )
    
    # 使用标签平滑交叉熵损失
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=TRAINING_CONFIG['smoothing']):
            super(LabelSmoothingCrossEntropy, self).__init__()
            self.smoothing = smoothing

        def forward(self, input, target):
            log_probs = torch.nn.functional.log_softmax(input, dim=-1)
            target = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
            target = target * (1 - self.smoothing) + (1 - target) * self.smoothing / (input.size(1) - 1)
            return (-target * log_probs).sum(dim=1).mean()
    
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG['n_epoch'],
        eta_min=TRAINING_CONFIG['eta_min']
    )
    
    return model, criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model():
    """训练模型的主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(PATH_CONFIG['output_dir'], exist_ok=True)
    
    # 创建日志目录
    if TENSORBOARD_AVAILABLE:
        os.makedirs(PATH_CONFIG['log_dir'], exist_ok=True)
        writer = SummaryWriter(log_dir=PATH_CONFIG['log_dir'])
    
    # 加载数据
    train_loader, val_loader, test_loader = load_data_and_create_loaders()
    
    # 创建模型和优化器
    model, criterion, optimizer, scheduler = create_model_and_optimizer(device)
    model.to(device)
    
    # 记录训练过程
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # 早停参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience = TRAINING_CONFIG['patience']
    
    print("Starting training...")
    
    for epoch in range(TRAINING_CONFIG['n_epoch']):
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['n_epoch']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 调整学习率
        scheduler.step()
        
        # 记录到tensorboard
        if TENSORBOARD_AVAILABLE:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Acc/Train', train_acc, epoch)
            writer.add_scalar('Acc/Val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(PATH_CONFIG['output_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses, 
        train_accuracies, val_accuracies,
        output_dir=PATH_CONFIG['output_dir']
    )
    
    if TENSORBOARD_AVAILABLE:
        writer.close()
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "="*50)
    print("Evaluating best model on test set...")
    print("="*50)
    
    # 在测试集上评估
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        label_names=EVALUATION_CONFIG['label_names'],
        save_results=True,
        output_dir=PATH_CONFIG['output_dir']
    )
    
    print(f"\nTraining completed. Best model saved to: {model_path}")
    
    return model, train_losses, val_losses


if __name__ == '__main__':
    trained_model, train_losses, val_losses = train_model()
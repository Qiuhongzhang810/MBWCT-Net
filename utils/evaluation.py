"""
MBWCT-Net 模型评估工具
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def evaluate_model(model, test_loader, device, label_names=None, save_results=True, output_dir='results'):
    """
    评估模型性能
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器
    :param device: 运行设备
    :param label_names: 标签名称列表
    :param save_results: 是否保存结果
    :param output_dir: 结果保存目录
    :return: 评估指标字典
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # 计算各类别的精确率、召回率和F1分数
    class_precision = precision_score(all_targets, all_preds, average=None)
    class_recall = recall_score(all_targets, all_preds, average=None)
    class_f1 = f1_score(all_targets, all_preds, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 打印详细的分类报告
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=label_names))
    
    # 绘制混淆矩阵
    if save_results:
        cm = confusion_matrix(all_targets, all_preds)
        plot_confusion_matrix(cm, label_names, output_dir)
    
    return metrics


def plot_confusion_matrix(cm, label_names=None, output_dir='results', filename='confusion_matrix.png'):
    """
    绘制并保存混淆矩阵
    :param cm: 混淆矩阵
    :param label_names: 标签名称列表
    :param output_dir: 输出目录
    :param filename: 文件名
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=label_names, 
        yticklabels=label_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 确保输出目录存在
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Confusion matrix saved to: {filepath}")
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accuracies=None, val_accuracies=None, output_dir='results'):
    """
    绘制训练曲线
    :param train_losses: 训练损失
    :param val_losses: 验证损失
    :param train_accuracies: 训练准确率
    :param val_accuracies: 验证准确率
    :param output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线（如果提供了的话）
    if train_accuracies and val_accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath)
    print(f"Training curves saved to: {filepath}")
    plt.close()


def calculate_model_complexity(model):
    """
    计算模型复杂度（参数量和FLOPs）
    :param model: PyTorch模型
    :return: 参数量和FLOPs
    """
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    result = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {result['non_trainable_params']:,}")
    
    return result


def compare_models(models_dict, test_loader, device, metric='f1'):
    """
    比较不同模型的性能
    :param models_dict: 模型字典，键为模型名称，值为模型实例
    :param test_loader: 测试数据加载器
    :param device: 设备
    :param metric: 用于比较的指标
    :return: 性能比较结果
    """
    results = {}
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        model.to(device)
        
        # 临时评估模型
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        if metric == 'f1':
            score = f1_score(all_targets, all_preds, average='weighted')
        elif metric == 'accuracy':
            score = accuracy_score(all_targets, all_preds)
        elif metric == 'precision':
            score = precision_score(all_targets, all_preds, average='weighted')
        elif metric == 'recall':
            score = recall_score(all_targets, all_preds, average='weighted')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results[name] = score
        print(f"{name} {metric}: {score:.4f}")
    
    # 排序并返回结果
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    print("\nModel Comparison Results:")
    for name, score in sorted_results.items():
        print(f"{name}: {score:.4f}")
    
    return sorted_results
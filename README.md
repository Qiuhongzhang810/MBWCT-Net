# MBWCT-Net: Multi-Branch Wavelet-CNN Transformer Network

## 简介

MBWCT-Net（Multi-Branch Wavelet-CNN Transformer Network）是一种用于多通道一维信号分类的深度学习模型。该模型结合了小波变换、卷积神经网络和Transformer架构，特别适用于气体传感器数据分析。

## 特性

- **多分支架构**: 使用多个并行分支处理输入信号的不同方面
- **小波变换**: 利用小波变换提取时频域特征
- **注意力机制**: 集成了ECA注意力和轻量级交叉注意力机制
- **MSAT-Former**: 多尺度自适应Transformer，融合局部和全局信息
- **灵活配置**: 可通过配置文件轻松修改模型参数

## 项目结构

```
MBWCT-Net/
├── algorithm/
│   ├── MBWCT-Net.py          # 核心模型定义
│   └── test_MBWCT-Net.py     # 测试脚本
├── models/                   # 其他模型实现
├── utils/
│   ├── data_loader.py        # 数据加载工具
│   └── evaluation.py         # 评估工具
├── config.py                 # 项目配置
├── train_mbwct.py            # 训练脚本
├── data_process.py           # 数据预处理
├── requirements.txt          # 依赖包
├── README.md                 # 项目说明
└── wood_gass_data_demo/      # 示例数据
    ├── Eucalyptus/
    ├── Rosewood/
    └── Sandalwood/
```

## 安装

1. 克隆项目：

```bash
git clone <repository-url>
cd MBWCT-Net
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python train_mbwct.py
```

### 自定义配置

修改 [config.py](file:///c:/Users/22209/Desktop/MBWCT-Net/config.py) 文件以调整模型参数和训练配置。

## 模型组件

### MSWPN
多尺度小波并行网络，提取信号的时频域特征。

### ECAAttention
高效通道注意力机制，专注于通道间的相互依赖关系。

### MSAT-Former
多尺度自适应Transformer，融合局部窗口注意力和全局自注意力。

### LightCrossAttention
轻量化的交叉注意力机制，用于多分支间的信息交互。

## 数据格式

输入数据应为三维张量 `(N, C, L)`，其中：
- `N`: 样本数量
- `C`: 通道数（默认8）
- `L`: 序列长度

标签应为一维张量 `(N,)`，包含类别索引。

## 配置选项

### 模型配置 ([MODEL_CONFIG](file:///c:/Users/22209/Desktop/MBWCT-Net/config.py#L6-L21))
- `in_channels`: 输入通道数
- `n_len_seg`: 每段长度
- `n_classes`: 分类数
- `num_branches`: 分支数量
- `use_msat_former`: 是否使用MSAT-Former

### 训练配置 ([TRAINING_CONFIG](file:///c:/Users/22209/Desktop/MBWCT-Net/config.py#L23-L33))
- `learning_rate`: 学习率
- `n_epoch`: 训练轮数
- `batch_size`: 批次大小
- `patience`: 早停耐心值

## 评估

模型评估包括：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数
- 混淆矩阵可视化

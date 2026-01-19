"""
MBWCT-Net 项目配置文件
用于管理所有超参数和配置选项
"""

# 模型配置
MODEL_CONFIG = {
    'in_channels': 8,      # 输入通道数
    'n_len_seg': 168,      # 每段长度
    'n_classes': 3,        # 分类数
    'num_branches': 5,     # 分支数量
    'wave': 'db4',         # 小波类型
    'J': 2,                # 小波分解层数
    'verbose': False,      # 是否打印详细信息
    'use_wavelet': True,   # 是否使用小波变换
    'use_cross_attn': True, # 是否使用交叉注意力
    'use_mlca': False,     # 是否使用MLCA
    'use_residual': False, # 是否使用残差连接
    'use_msat_former': True # 是否使用MSAT-Former
}

# 训练配置
TRAINING_CONFIG = {
    'smoothing': 0.038,      # 标签平滑参数
    'learning_rate': 0.0005, # 学习率
    'weight_decay': 0.00006, # 权重衰减
    'n_epoch': 150,          # 训练轮次
    'patience': 15,          # 早停耐心值
    'eta_min': 1e-6,         # 余弦退火最小学习率
    'batch_size': 32,        # 批次大小
    'test_size': 0.2         # 测试集比例
}

# 路径配置
PATH_CONFIG = {
    'data_path': './data/augmented_data.npy',  # 数据路径
    'output_dir': 'output/mbwct_net',          # 输出目录
    'log_dir': 'logs/mbwct_net'               # 日志目录
}

# 设备配置
DEVICE_CONFIG = {
    'device': 'cuda' if True else 'cpu'  # 将在运行时动态确定
}

# 评估配置
EVALUATION_CONFIG = {
    'label_names': ['Eucalyptus', 'Rosewood', 'Sandalwood']  # 类别名称
}
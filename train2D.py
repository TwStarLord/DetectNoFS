# 1.读取config2D.yaml文件，初始化模型
# 2.读取数据集，并划分训练集和验证集
# 3.训练模型，并保存模型
# 4.验证模型，并保存验证结果

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset

from utils.config_util import load_config
from utils.register import ModuleRegister
from models import TcRadar2D_pytorch

# —— 1. 读取并解析配置文件 —— #
config_path = 'config/config2D.yaml'
config      = load_config(config_path)
blocks_cfg  = config['blocks']

# —— 2. 初始化模块注册器 —— #
mreg = ModuleRegister(blocks_cfg)

# —— 3. 构造模型 —— #
#    将注册器、隐层维度和类别数一起传入，模型内部再做具体实例化
hidden_dim  = 96
num_classes = 5

model = TcRadar2D_pytorch(
    module_register=mreg,
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=5,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
).to('cuda' if torch.cuda.is_available() else 'cpu')


# —— 4. 剩余的训练/验证流程与以前相同 —— #
# （此处省略数据加载、训练循环等常规代码）


# 实例化并移动到设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = Tc_Radar2D(ca, cls_head, difb, fusion, upsample).to(device)

# —— 接下来可按常规训练流程：加载数据、定义损失/优化器、训练/验证、保存 Checkpoint —— #
# … 以下略 …



# 移动模型到计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectNoFS2DModel(
    ca=ca_module,
    cls_head=cls_head_module,
    difb=difb_module,
    fusion=fusion_module,
    upsample=upsample_module
).to(device)

# 准备数据集（示例：使用随机数据，实际项目请替换为真实数据集）
dataset = TensorDataset(
    torch.randn(100, 3, 224, 224),
    torch.randint(0, 10, (100,))
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# 根据配置的模块组合动态生成日志目录
module_combination_name = f"{config['DIFB']}_{config['feature_fusion']}_{config['upsample']}"
log_dir = os.path.join('logs', module_combination_name)
os.makedirs(log_dir, exist_ok=True)

# 训练循环
for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 简单验证过程（计算验证集准确率作为示例）
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total if total > 0 else 0
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {total_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # 保存检查点，文件名包含模块组合和轮数
    ckpt_filename = f"ckpt_epoch{epoch + 1}.pth"
    ckpt_path = os.path.join(log_dir, ckpt_filename)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

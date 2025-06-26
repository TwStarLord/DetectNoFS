import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

# 定义图像预处理：缩放到224×224，灰度转3通道，转换为Tensor并归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),                         # 统一尺寸
    transforms.Grayscale(num_output_channels=3),           # 转为3通道灰度图（若原为RGB，可省略）
    transforms.ToTensor(),                                 # 转为0-1张量
    transforms.Normalize(mean=[0.5,0.5,0.5],               # 归一化（可根据模型预训练时使用的均值方差调整）
                         std=[0.5,0.5,0.5])
])

# 加载整个数据集，ImageFolder要求目录下每个子文件夹名为类别名
data_dir = "dataset/train"  # 或整个数据集根目录，视具体情况
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 获取样本数和对应标签列表
num_samples = len(dataset)
targets = dataset.targets  # 整数标签列表

# 使用train_test_split做分层划分，先划分测试集（20%），再从剩下划分验证集（20% of remaining）
indices = list(range(num_samples))
# 1) 划分出测试集20%
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=targets)
# 2) 从训练集中划分验证集20%
train_targets = [targets[i] for i in train_idx]
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=train_targets)

# 创建Subset数据集
train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx)
test_set  = Subset(dataset, test_idx)

# 构造DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4)

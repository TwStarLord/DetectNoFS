import os
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset

from models.TcRadar2D_pytorch import Tc_Radar2D
from utils.config_util import load_config
from utils.register import ModuleRegister

# 导入绘图与评估库
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# —— 数据加载函数 —— #
def load_data(batch_size: int, num_classes: int):
    dataset = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, num_classes, (100,))
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

# —— 模型构建函数 —— #
def build_model(mreg, model_cfg: dict, device: torch.device):
    # 处理 YAML 中可能被解析为字符串的列表结构
    def parse_tuple(value):
        if isinstance(value, str):
            return tuple(ast.literal_eval(value))
        return tuple(value)

    layers = parse_tuple(model_cfg['layers'])
    heads = parse_tuple(model_cfg['heads'])
    downscaling = parse_tuple(model_cfg['downscaling_factors'])

    # 确保数值型配置正确
    hidden_dim = int(model_cfg.get('hidden_dim', 96))
    channels = int(model_cfg.get('channels', 3))
    num_classes = int(model_cfg['num_classes'])
    head_dim = int(model_cfg.get('head_dim', 32))
    window_size = int(model_cfg.get('window_size', 7))
    relative_pos_embedding = bool(model_cfg.get('relative_pos_embedding', True))

    model = Tc_Radar2D(
        module_register=mreg,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        channels=channels,
        num_classes=num_classes,
        head_dim=head_dim,
        window_size=window_size,
        downscaling_factors=downscaling,
        relative_pos_embedding=relative_pos_embedding
    ).to(device)
    return model

# —— 单个训练周期函数 —— #
def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

# —— 验证并收集预测结果函数 —— #
def validate(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0
    return acc, y_true, y_pred

# —— 绘图与指标记录函数 —— #
def log_and_plot_metrics(y_true: list, y_pred: list, classes: list, save_dir: str):
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close(fig)

# —— 主函数 —— #
def main():
    # 1. 读取配置
    config = load_config('config/config2D.yaml')
    blocks_cfg = config['blocks']
    model_cfg = config['model']

    # 2. 解析数值型配置，避免字符串类型
    batch_size = int(model_cfg['batch_size'])
    num_epochs = int(model_cfg['num_epochs'])
    learning_rate = float(model_cfg['learning_rate'])
    num_classes = int(model_cfg['num_classes'])

    # 3. 初始化模块注册器与设备
    mreg = ModuleRegister(blocks_cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. 构造模型
    model = build_model(mreg, model_cfg, device)

    # 5. 准备数据
    train_loader, val_loader = load_data(batch_size, num_classes)

    # 6. 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 7. 日志目录
    combo = f"{blocks_cfg['DIFB']}_{blocks_cfg['feature_fusion']}_{blocks_cfg['upsample']}"
    log_dir = os.path.join('logs', combo)
    os.makedirs(log_dir, exist_ok=True)

    # 8. 训练与验证循环
    train_losses, val_accs = [], []
    all_val_true, all_val_pred = [], []
    classes = [str(i) for i in range(num_classes)]

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc, y_true, y_pred = validate(model, val_loader, device)
        train_losses.append(loss)
        val_accs.append(acc)
        all_val_true, all_val_pred = y_true, y_pred

        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Val Acc: {acc:.4f}")

        # 保存 checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'device': str(device)
        }
        ckpt_path = os.path.join(log_dir, f"ckpt_epoch{epoch}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    # 9. 绘制训练曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss Curve')
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, num_epochs + 1), val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(log_dir, 'accuracy_curve.png'))
    plt.close()

    # 10. 输出分类报告与混淆矩阵
    log_and_plot_metrics(all_val_true, all_val_pred, classes, log_dir)

if __name__ == '__main__':
    main()

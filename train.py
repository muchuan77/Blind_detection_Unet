#train.py
import os
import logging

# 首先设置环境变量以减少tensorflow的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 仅显示warning和error

# 配置Python的全局日志级别为WARNING，减少日志噪音
logging.basicConfig(level=logging.WARNING)

import datetime
import json

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from nets.unet import UNet
from utils.dataset import SegmentationDataset
from utils.utils import get_optimizer, get_scheduler, train_one_epoch, LossHistory, evaluate
import argparse

# 首先设置环境变量以减少tensorflow的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 仅显示warning和error

# 配置Python的全局日志级别为WARNING，减少日志噪音
logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet model with different backbones.")
    parser.add_argument('--mode', type=str, default='config', choices=['config', 'select', 'list'],
                        help='Training mode: use "config" for config.json, "select" to select a backbone, "list" to iterate through the backbone list.')
    parser.add_argument('--backbone', type=str, default=None, help='Backbone name for the "select" mode.')
    return parser.parse_args()


# 从文件中读取图像列表
def read_image_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


# 定义您想要自动训练的backbone列表
backbones = ["vgg", "resnet"]


# 训练并评估模型的函数
def train_and_evaluate(backbone, config, device):
    # 读取配置文件，并根据当前迭代的backbone更新配置
    config['backbone'] = backbone
    print(f"正在使用 {backbone} 作为backbone进行训练")

    # 初始化模型
    model = UNet(num_classes=config['num_classes'], backbone=backbone, pretrained=True).to(device)

    # 加载数据
    train_image_list = read_image_list(os.path.join(config['dataset_path'], 'Index/train.txt'))
    val_image_list = read_image_list(os.path.join(config['dataset_path'], 'Index/val.txt'))
    train_dataset = SegmentationDataset(image_list=train_image_list, dataset_path=config['dataset_path'], target_size=(512, 512))
    val_dataset = SegmentationDataset(image_list=val_image_list, dataset_path=config['dataset_path'], target_size=(512, 512))
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 定义criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)  # 确保损失函数也移到了正确的设备
    scaler = GradScaler()  # 初始化梯度缩放器

    # 设置优化器和学习率调度器
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # 读取训练日志路径
    train_log_dir = str(config['train_log_path'])
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(train_log_dir, backbone, current_time)
    os.makedirs(train_log_dir, exist_ok=True)  # 确保目录存在

    # 初始化SummaryWriter和LossHistory
    writer = SummaryWriter(log_dir=train_log_dir)
    loss_history = LossHistory(train_log_dir)

    best_train_loss = 0.0
    best_train_model_path = ""
    best_val_iou = 0.0
    best_val_model_path = ""

    # 开始训练循环
    for epoch in range(config['epochs']):
        print(f'开始 Epoch {epoch + 1}/{config["epochs"]}')

        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            # 使用autocast上下文管理器进行自动混合精度训练
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # 记录训练损失和学习率
            if batch_idx % 100 == 99:  # 每100个batch记录一次
                writer.add_scalar('Loss/train', running_loss / 100, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],
                                  epoch * len(train_loader) + batch_idx)
                running_loss = 0.0

        # 从 train_one_epoch 获取平均损失
        train_avg_loss = train_one_epoch(model, device, train_loader, optimizer, epoch)
        writer.add_scalar(f'Loss/Train_{backbone}', train_avg_loss, epoch)
        loss_history.add_loss(train_avg_loss)
        # 更新最佳训练loss模型
        if train_avg_loss < best_train_loss:
            best_train_loss = train_avg_loss
            best_train_model_path = os.path.join(config['best_model_path'], f'best_train_model_{backbone}.pth')
            torch.save(model.state_dict(), best_train_model_path)

        # 调整学习率
        scheduler.step()

        # 在每个epoch结束后评估模型(val)
        val_iou_scores = evaluate(model, device, val_loader, config['num_classes'],epoch)
        val_avg_iou = np.nanmean(val_iou_scores)
        writer.add_scalar(f'IoU/Val_{backbone}', val_avg_iou, epoch + 1)
        if val_avg_iou > best_val_iou:
            best_val_iou = val_avg_iou
            best_val_model_path = os.path.join(config['best_model_path'], f'best_val_model_{backbone}.pth')
            torch.save(model.state_dict(), best_val_model_path)

        # 每10个epoch保存一次模型
        if (epoch+1) % 10 == 0:
            model_save_path = os.path.join(train_log_dir, f'model_{backbone}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'模型已保存至 {model_save_path}')

            # 可选：记录参数和梯度直方图
            for name, param in model.named_parameters():
                writer.add_histogram('Parameters/' + name, param, epoch)
                if param.grad is not None:
                    writer.add_histogram('Gradients/' + name, param.grad, epoch)

            # 可选：记录特定层的特征图和预测结果示例
            if epoch % 10 == 0:  # 每10个epoch做一次
                with torch.no_grad():
                    inputs, labels = next(iter(val_loader))
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).unsqueeze(1)

                    # 选择记录的图像数量
                    num_images = min(inputs.size(0), 5)

                    # 将输入、标签和预测转换为图像格式并记录
                    images_to_show = torch.cat([inputs[:num_images], labels[:num_images], preds[:num_images]], 0)
                    grid = make_grid(images_to_show, nrow=num_images, normalize=True, scale_each=True)
                    writer.add_image(f'Comparison/{backbone}', grid, epoch)

    print(f'train最佳模型已保存至 {best_train_model_path}, loss: {best_train_loss:.4f}')
    print(f'val最佳模型已保存至 {best_val_model_path}, IoU: {best_val_iou:.4f}')
    val_iou_scores = evaluate(model, device, val_loader, config['num_classes'], epoch)
    val_avg_iou = np.nanmean(val_iou_scores)
    writer.add_scalar(f'IoU/Val_{backbone}', val_avg_iou, epoch)

    # 训练结束后保存损失历史
    loss_history.save()

    writer.close()  # 关闭 TensorBoard 日志记录器


# 主函数，从这里开始执行
def main():
    args = parse_args()  # 先前添加的解析命令行参数的函数

    # 加载配置文件
    with open('config.json', 'r') as file:
        config = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get('seed', 42))
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # 根据命令行参数决定训练模式
    if args.mode == 'config':
        # 使用 config.json 中的 backbone
        backbones = [config['backbone']]
    elif args.mode == 'list':
        # 遍历所有列出的 backbones
        backbones = ["vgg", "resnet"]  # 或者任何你想遍历的backbone列表
    elif args.mode == 'select':
        # 选择列表中的单个 backbone
        if args.backbone in ["vgg", "resnet"]:  # 校验用户输入是否有效
            backbones = [args.backbone]
        else:
            raise ValueError("Unsupported backbone. Please choose either 'vgg' or 'resnet'.")
    else:
        raise ValueError("Unsupported mode. Please use --mode to specify 'config', 'select', or 'list'.")

    # 根据选择的backbone列表进行模型训练
    for backbone in backbones:
        print(f"Training with backbone: {backbone}")
        train_and_evaluate(backbone, config, device)


if __name__ == "__main__":
    main()


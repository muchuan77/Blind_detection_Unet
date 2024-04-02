#utils.py
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import calculate_iou


# 图像预处理函数
def cvtColor(image):
    # 如果图像不是三通道，转换为RGB
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    # 调整图像尺寸并保持纵横比
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def preprocess_input(image):
    # 图像归一化
    image = np.array(image) / 255.0
    return image


# 获取优化器和学习率调度器
def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    # 使用Adam优化器
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer):
    # 学习率每7个步骤下降10倍
    return StepLR(optimizer, step_size=7, gamma=0.1)


# 训练一个epoch
def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # 假设是分类任务
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    print(f'Epoch: {epoch+1}, Average Train_Loss: {avg_loss:.4f}')
    return avg_loss


# 评估模型
def evaluate(model, device, val_loader, num_classes, epoch):
    model.eval()
    total_iou = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # 计算IoU
            iou_list = calculate_iou(outputs, target, num_classes)
            total_iou.extend(iou_list)

    # 过滤掉 NaN 值
    total_iou = [x for x in total_iou if not np.isnan(x)]

    # 如果没有有效的 IoU 值，则返回 NaN
    if not total_iou:
        return float('nan')

    avg_iou = np.mean(total_iou)
    print(f'Epoch: {epoch+1}, Average Val_IoU: {avg_iou:.4f}')
    return avg_iou


# 设置种子，确保实验可重复
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 工作初始化函数，用于多进程时设置不同的随机种子
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


# 显示配置信息
def show_config(**kwargs):
    print('Configurations:')
    for key, value in kwargs.items():
        print(f'{key}: {value}')


# 根据backbone下载预训练权重
def download_weights(backbone):
    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    }
    # 指定模型存储的目录
    model_dir = "./model_data"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 下载并加载预训练权重
    if backbone in download_urls:
        state_dict = load_state_dict_from_url(download_urls[backbone], model_dir=model_dir, progress=True)
        print(f"{backbone}的预训练权重已下载并加载。")
    else:
        print(f"不支持的backbone类型: {backbone}。请检查是否正确或支持的列表中添加。")


# 损失历史记录工具
class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def save(self):
        # 保存损失历史到文件
        loss_path = os.path.join(self.log_dir, "loss_history.npy")
        np.save(loss_path, np.array(self.losses))
        print(f"损失历史已保存至 {loss_path}")

# 注意: 上述代码示例中的 `load_state_dict_from_url` 需要联网下载预训练权重。确保在可以联网的环境下执行。
# 另外, 在使用 `np.save` 保存损失历史时, 确保 `self.log_dir` 目录已存在。


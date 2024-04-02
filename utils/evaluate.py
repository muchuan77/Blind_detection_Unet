#evaluate.py
import json
import os

import torch
from torch.utils.data import DataLoader
from nets.unet import UNet
from utils.dataset import SegmentationDataset
from utils.metrics import evaluate


def read_image_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


def main():
    # 读取配置文件
    with open('config.json', 'r') as file:  # 确保这是您config.json文件的正确路径
        config = json.load(file)

    # 从配置文件获取参数
    num_classes = config['num_classes']
    model_path = config['model_path']
    dataset_path = config['dataset_path']

    # 使用此函数将图像标识符读入列表
    val_image_list = read_image_list(os.path.join(dataset_path, 'Index/val.txt'))

    # 创建数据加载器
    val_dataset = SegmentationDataset(image_list=val_image_list, dataset_path=dataset_path)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=num_classes).to(device)

    # 加载训练好的模型
    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    # 调用评估函数
    evaluate(model, device, val_loader,num_classes)


if __name__ == "__main__":
    main()

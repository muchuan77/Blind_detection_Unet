# #test.py
# import argparse
# import os
# import json
# import torch
# import numpy as np
# from PIL import Image
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from nets.unet import UNet
# from utils.dataset import SegmentationDataset
# from utils.metrics import calculate_iou
#
#
# # 定义函数用于读取图像列表
# def read_image_list(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file]
#
#
# # 定义模型加载函数
# def load_model(model_path, device, num_classes, backbone):
#     model = UNet(num_classes=num_classes, backbone=backbone, pretrained=False).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     return model
#
#
# # 定义预测函数
# def predict(model, device, data_loader):
#     model.eval()
#     total_iou = 0
#     count = 0
#     for data, target in tqdm(data_loader, desc="Predicting"):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         # 计算IoU并累加
#         iou = calculate_iou(output, target, num_classes=2)
#         total_iou += np.nanmean(iou)  # 忽略nan值
#         count += 1
#     avg_iou = total_iou / count
#     print(f"Average IoU: {avg_iou:.4f}")
#     return avg_iou
#
#
# # 定义保存预测结果的函数
# def save_predictions(model, device, data_loader, save_dir):
#     model.eval()
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for i, (data, _) in enumerate(tqdm(data_loader, desc="Saving predictions")):
#         data = data.to(device)
#         with torch.no_grad():
#             output = model(data)
#             pred = torch.argmax(output, dim=1).cpu().numpy()
#         pred_img = Image.fromarray(pred[0].astype(np.uint8))
#         pred_img.save(os.path.join(save_dir, f'pred_{i}.png'))
#
#
# # 新添加的命令行参数处理
# def get_args():
#     parser = argparse.ArgumentParser(description="Test the model on a dataset.")
#     parser.add_argument("--backbone", type=str, choices=["vgg", "resnet"], help="Choose the model backbone.")
#     parser.add_argument("--data_type", type=str, choices=["train", "val"], help="Choose the dataset type for testing.")
#     args = parser.parse_args()
#     return args
#
#
# from torch.utils.tensorboard import SummaryWriter
#
# def main():
#     args = get_args()  # 获取命令行参数
#
#     with open('config.json', 'r') as file:
#         config = json.load(file)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = os.path.join(config['best_model_path'], f'best_{args.data_type}_model_{args.backbone}.pth')
#
#     # 初始化TensorBoard SummaryWriter
#     test_log_dir = os.path.join(config['test_log_path'], args.backbone, args.data_type)
#     writer = SummaryWriter(log_dir=test_log_dir)
#
#     # 确保模型路径存在
#     if not os.path.exists(model_path):
#         print(f"Model {model_path} not found.")
#         return
#
#     # 加载模型
#     model = load_model(model_path, device, config['num_classes'], args.backbone)
#
#     # 准备数据
#     dataset_path = os.path.join(config['dataset_path'], 'Index', f'{args.data_type}.txt')
#     image_list = read_image_list(dataset_path)
#     dataset = SegmentationDataset(image_list=image_list, dataset_path=config['dataset_path'], target_size=(512, 512))
#     data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
#
#     # 进行预测并计算平均IoU
#     avg_iou = predict(model, device, data_loader)
#     print(f"Average IoU for the {args.data_type} set is {avg_iou:.4f}")
#     # 将平均IoU记录到TensorBoard
#     writer.add_scalar("Average IoU", avg_iou, 0)
#
#     # 保存预测结果
#     save_dir = os.path.join(config['prediction_path'], args.data_type)
#     save_predictions(model, device, data_loader, save_dir)
#     print(f"Predictions saved in {save_dir}")
#
#     writer.close()
#
#
# if __name__ == "__main__":
#     main()
#
#
#
# if __name__ == "__main__":
#     main()


import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nets.unet import UNet
from utils.dataset import SegmentationDataset
from utils.metrics import calculate_iou


# 从文件中读取图像列表
def read_image_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


# 定义模型加载函数
def load_model(model_path, device, num_classes, backbone):
    model = UNet(num_classes=num_classes, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


# 定义预测函数
def predict(model, device, data_loader):
    model.eval()
    total_iou = 0
    count = 0
    for data, target in tqdm(data_loader, desc="Predicting"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        iou = calculate_iou(output, target, num_classes=2)
        total_iou += np.nanmean(iou)  # 忽略nan值
        count += 1
    avg_iou = total_iou / count
    print(f"Average IoU: {avg_iou:.4f}")
    return avg_iou


# 定义保存预测结果的函数
def save_predictions(model, device, data_loader, save_dir):
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (data, _) in enumerate(tqdm(data_loader, desc="Saving predictions")):
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            pred = torch.argmax(output, dim=1).cpu().numpy()
        pred_img = Image.fromarray(pred[0].astype(np.uint8))
        pred_img.save(os.path.join(save_dir, f'pred_{i}.png'))


# 新添加的命令行参数处理
def get_args():
    parser = argparse.ArgumentParser(description="Test the model on a dataset.")
    parser.add_argument("--backbone", type=str, choices=["vgg", "resnet"], help="Choose the model backbone.")
    parser.add_argument("--data_type", type=str, choices=["train", "val"], help="Choose the dataset type for testing.")
    return parser.parse_args()


def main():
    args = get_args()  # 获取命令行参数

    with open('config.json', 'r') as file:
        config = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config['best_model_path'], f'best_{args.data_type}_model_{args.backbone}.pth')

    # 初始化TensorBoard SummaryWriter
    test_log_dir_str = str(os.path.join(config['test_log_path'], args.backbone, args.data_type))
    os.makedirs(test_log_dir_str, exist_ok=True)
    writer = SummaryWriter(log_dir=test_log_dir_str)

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    model = load_model(model_path, device, config['num_classes'], args.backbone)

    dataset_path = os.path.join(config['dataset_path'], 'Index', f'{args.data_type}.txt')
    image_list = read_image_list(dataset_path)
    dataset = SegmentationDataset(image_list=image_list, dataset_path=config['dataset_path'], target_size=(512, 512))
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    avg_iou = predict(model, device, data_loader)
    writer.add_scalar("Average IoU", avg_iou, 0)

    save_dir = os.path.join(config['prediction_path'], args.data_type)
    save_predictions(model, device, data_loader, save_dir)
    print(f"Predictions saved in {save_dir}")

    writer.close()


if __name__ == "__main__":
    main()

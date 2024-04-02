#dataset.py
from utils.utils import cvtColor, resize_image, preprocess_input

import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class SegmentationDataset(Dataset):
    def __init__(self, image_list, dataset_path, target_size=(512, 512), transform=None):
        #transform预设
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) if transform is None else transform

        self.dataset_path = dataset_path
        self.image_list = image_list  # 直接使用传入的图像列表
        self.target_size = target_size
        # self.transform = transform

    def __len__(self):
        return len(self.image_list)

    #无transform预设
    # def __getitem__(self, idx):
    #     # 图像和标签的文件路径
    #     img_file = os.path.join(self.dataset_path, "JPEGImages", self.image_list[idx] + ".jpg")
    #     label_file = os.path.join(self.dataset_path, "SegmentationClassPNG", self.image_list[idx] + ".png")
    #
    #     # 加载图像和标签
    #     image = Image.open(img_file)
    #     label = Image.open(label_file)
    #
    #     # 如果有必要，可以在这里添加任何转换处理
    #     if self.transform is not None:
    #         image = self.transform(image)
    #         label = self.transform(label)
    #
    #     # 调整图像和标签的尺寸
    #     image = resize_image(image, self.target_size)
    #     label = label.resize(self.target_size, resample=Image.NEAREST)
    #
    #     # 预处理
    #     image = np.array(image, dtype=np.float32)
    #     label = np.array(label, dtype=np.int64)
    #
    #     # 转换为tensor
    #     image = transforms.ToTensor()(image)
    #     label = torch.from_numpy(label)
    #
    #     return image, label

    def __getitem__(self, idx):
        img_file = os.path.join(self.dataset_path, "JPEGImages", self.image_list[idx] + ".jpg")
        label_file = os.path.join(self.dataset_path, "SegmentationClassPNG", self.image_list[idx] + ".png")
        image = Image.open(img_file).convert("RGB")
        label = Image.open(label_file)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)

        label = TF.resize(label, self.target_size, interpolation=InterpolationMode.NEAREST)
        label = np.array(label, dtype=np.int64)  # 直接使用numpy来处理标签转换为数组
        label = torch.as_tensor(label)  # 使用torch.as_tensor来将numpy数组转换为Tensor

        return image, label
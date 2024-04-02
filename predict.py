import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import argparse
import cv2
from nets.unet import UNet  # 确保这个路径正确指向了你的UNet实现


def load_model(model_path, device, num_classes, backbone):
    """
    加载预训练的UNet模型。

    参数:
    - model_path: 预训练模型的文件路径。
    - device: 指定模型运行的设备（'cuda'或'cpu'）。
    - num_classes: 分类任务的类别数量。
    - backbone: 指定的backbone名称（'vgg'或'resnet'）。

    返回:
    - 加载的模型实例。
    """
    model = UNet(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, device, image_tensor):
    """
    使用加载的模型对单个图像进行预测。

    参数:
    - model: 加载的模型实例。
    - device: 模型运行的设备。
    - image_tensor: 图像数据的tensor形式。

    返回:
    - 预测结果的numpy数组。
    """
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return prediction


def process_frame(frame, model, device):
    """
    处理视频帧，转换为模型可接受的输入，进行预测。

    参数:
    - frame: 视频帧的numpy数组。
    - model: 加载的模型实例。
    - device: 模型运行的设备。

    返回:
    - 对该帧的预测结果。
    """
    # 将帧转换为PIL图像，进行预处理，然后转换为tensor
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    prediction = predict_image(model, device, image_tensor)
    return prediction


def handle_video(input_path, model, device):
    """
    处理视频文件或实时监视器输入。

    参数:
    - input_path: 视频文件的路径或摄像头设备索引。
    - model: 加载的模型实例。
    - device: 模型运行的设备。
    """
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        prediction = process_frame(frame, model, device)
        # 可视化预测结果
        # 示例: 显示原始视频帧，你可以根据需要添加预测结果的可视化
        cv2.imshow('Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("请指定输入路径（图像、视频文件路径或 'camera' 以使用摄像头）: ")
    input_path = input()

    print("请选择模型类型（'val' 或 'train'）: ")
    model_type = input()

    print("请选择backbone类型（'vgg' 或 'resnet'）: ")
    backbone = input()

    with open('config.json') as file:
        config = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['num_classes']
    # 注意这里路径构建的方式，确保符合你的文件命名和路径结构
    model_path = f"{config['best_model_path']}/best_{model_type}_model_{backbone}.pth"

    model = load_model(model_path, device, num_classes, backbone)

    if input_path in ['camera', '0']:
        handle_video(0, model, device)
    elif input_path.endswith('.mp4') or input_path.endswith('.avi'):
        handle_video(input_path, model, device)
    else:
        # 处理单张图片
        original_image = Image.open(input_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 调整图像尺寸以匹配模型预期的输入
            transforms.ToTensor(),
        ])
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        prediction = predict_image(model, device, image_tensor)

        # 将预测结果转换回图像形式并保存
        pred_image = Image.fromarray((prediction * 255 / num_classes).astype(np.uint8))
        pred_image = pred_image.resize(original_image.size, Image.NEAREST)  # 调整回原始尺寸
        output_path = "predicted_image.png"
        pred_image.save(output_path)
        print(f"预测结果已保存至 {output_path}")


if __name__ == "__main__":
    main()


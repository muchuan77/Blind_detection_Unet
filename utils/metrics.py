#metrics.py
import torch


def calculate_iou(pred, target, num_classes):
    iou_list = []
    pred = torch.argmax(pred, dim=1)  # 假设pred是模型的原始输出，需要使用argmax获得最大概率的类别索引
    for cls in range(num_classes):  # 遍历每一个类别
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).sum()  # 交集
        union = pred_inds.sum() + target_inds.sum() - intersection  # 并集
        if union == 0:
            iou_list.append(float('nan'))  # 避免除以0
        else:
            iou_list.append(float(intersection) / float(max(union, 1)))
    return iou_list


def evaluate(model, device, data_loader, num_classes):
    model.eval()
    total_iou = 0.0
    num_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            iou_list = calculate_iou(output, target, num_classes)
            # 转换iou_list为Tensor
            iou_tensor = torch.tensor(iou_list, device=device)
            # 过滤NaN值
            valid_iou_tensor = iou_tensor[~torch.isnan(iou_tensor)]
            total_iou += valid_iou_tensor.sum().item()
            num_samples += valid_iou_tensor.size(0)
    avg_iou = total_iou / max(num_samples, 1)  # 避免除以0
    print(f'Average IoU: {avg_iou:.4f}')


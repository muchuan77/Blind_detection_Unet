#unet.py
import torch.nn as nn
from .vgg import VGG16
from .resnet import resnet50


class UNet(nn.Module):
    def __init__(self, num_classes, backbone='vgg', pretrained=True):
        super(UNet, self).__init__()
        if backbone == 'vgg':
            self.encoder = VGG16(pretrained=pretrained)
            filters = [64, 128, 256, 512, 512]
        elif backbone == 'resnet':
            # 注意这里调用 resnet50 函数，并传入预训练参数
            self.encoder = resnet50(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]  # 调整为ResNet的特征维度
        else:
            raise ValueError("Backbone should be 'vgg' or 'resnet'")
        # 解码器部分，这里需要根据编码器的特征层数量来设计
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            self.decoder_blocks.append(
                DecoderBlock(filters[i], filters[i - 1])
            )
        # 分类层
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    # def forward(self, x):
    #     # 编码器层输出
    #     features = self.encoder(x)
    #     # 假设features是一个包含多个特征图的列表
    #
    #     # 初始解码器输入为最深层特征图
    #     decoder_input = features[-1]
    #     for i in range(len(self.decoder_blocks)):
    #         # 逆序应用解码器块
    #         decoder_input = self.decoder_blocks[i](decoder_input)
    #         # 特征融合（示例），需要根据您的架构调整
    #         if i < len(features) - 1:
    #             # 使用上采样或其他方法调整features[i]的尺寸以匹配decoder_input
    #             # 然后融合这两个特征图
    #             decoder_input = decoder_input + features[-2 - i]  # 这里是简化的示例，可能需要调整
    #
    #     x = self.final_conv(decoder_input)
    #     return x

    def forward(self, x):
        # 编码器层输出
        features = self.encoder(x)
        # 初始解码器输入为最深层特征图
        decoder_input = features[-1]
        # 逆序应用解码器块
        for decoder_block, feature in zip(self.decoder_blocks, features[::-1][1:]):
            decoder_input = decoder_block(decoder_input)
            # 可选：特征融合
            # 如果你的目标是精确的分割任务，可能需要在这里融合编码器和解码器的特征
            # 注意：可能需要调整特征尺寸以确保它们可以被融合
        # 确保最终输出尺寸与目标匹配
        # 如果decoder_input的尺寸不是512x512，这里需要再次上采样
        if decoder_input.shape[-2:] != (512, 512):
            decoder_input = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)(decoder_input)
        x = self.final_conv(decoder_input)
        return x


class DecoderBlock(nn.Module):
    # 定义解码器模块
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decode(x)

# 使用示例
# model = UNet(num_classes=21, backbone='vgg')

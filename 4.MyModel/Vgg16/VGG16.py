import torch
import torch.nn as nn
import torch.nn.functional as F


# conv的stride为1，padding为1
# Ø maxpool的size为2，stride为2


# 封装卷积和激活函数
import torchvision.models


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=1, padding=1, stride=1, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, **kwargs)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        return x


class Vgg16(nn.Module):
    def __init__(self, num_classes=6):  # 若num_classes> 512 需要修改最后一层LInear结构
        super(Vgg16, self).__init__()
        self.layer1 = nn.Sequential(BasicConv2d(in_channel=3, out_channel=64, kernel_size=3),
                                    BasicConv2d(in_channel=64, out_channel=64, kernel_size=3)
                                    )

        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicConv2d(in_channel=64, out_channel=128, kernel_size=3),
                                    BasicConv2d(in_channel=128, out_channel=128, kernel_size=3)
                                    )

        self.layer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicConv2d(in_channel=128, out_channel=256, kernel_size=3),
                                    BasicConv2d(in_channel=256, out_channel=256, kernel_size=3),
                                    BasicConv2d(in_channel=256, out_channel=256, kernel_size=3)
                                    )
        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicConv2d(in_channel=256, out_channel=512, kernel_size=3),
                                    BasicConv2d(in_channel=512, out_channel=512, kernel_size=3),
                                    BasicConv2d(in_channel=512, out_channel=512, kernel_size=3)
                                    )
        self.layer5 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicConv2d(in_channel=512, out_channel=512, kernel_size=3),
                                    BasicConv2d(in_channel=512, out_channel=512, kernel_size=3),
                                    BasicConv2d(in_channel=512, out_channel=512, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # 以50%概率失活神经元，防止过拟合
            nn.Linear(7 * 7 * 512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    VGGModel = Vgg16(num_classes=6)
    print(VGGModel)
torchvision.models.ResNet
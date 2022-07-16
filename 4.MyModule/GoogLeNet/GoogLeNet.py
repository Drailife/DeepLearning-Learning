import torch
import torch.nn as nn
import torch.nn.functional as F


# 将conv2d 和 relu 整合为一个模块，减少代码量
class BasicCov2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, padding=1,
                 stride=1, **kwargs):
        super(BasicCov2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride,
                              kernel_size=kernel_size, padding=padding, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


# 实现 inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, out_ch1x1, out_ch3x3x1, out_ch3x3x2, out_ch5x5x1, out_ch5x5x2, out_ch_pool):
        super(Inception, self).__init__()
        # branch one
        self.branch1 = BasicCov2d(in_channels, out_ch1x1, kernel_size=1)

        # branch two
        self.branch2 = nn.Sequential(
            BasicCov2d(in_channels, out_ch3x3x1, kernel_size=1),
            BasicCov2d(out_ch3x3x1, out_ch3x3x2, kernel_size=3, padding=1)
            # 保证输入输出大小相同
            # 计算公式为 (in_size - ker +2padding) / stride + 1 = out_size
        )

        # branch three
        self.branch3 = nn.Sequential(
            BasicCov2d(in_channels, out_ch5x5x1, kernel_size=1),
            BasicCov2d(out_ch5x5x1, out_ch5x5x2, kernel_size=5, padding=2)
        )

        # branch four
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicCov2d(in_channels, out_ch_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        sequential = [branch1, branch2, branch3, branch4]
        x = torch.cat(sequential, dim=1)  # 按照clannel维度进行连接
        # pytorch 中维度排列为【 batch_size, channel, height, width】
        return x


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.averagepool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicCov2d(in_channel, out_channels=128, kernel_size=1)  # output = [N , 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagepool(x)
        x = self.conv(x)
        x = F.dropout(x, 0.5, training=self.training)  # 神经元随机失活(50%)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class MyGoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weight=False):
        super(MyGoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicCov2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 若计算的size为小数，若ceil_mode=True 向上取整

        self.conv2 = BasicCov2d(64, 64, kernel_size=1)
        self.conv3 = BasicCov2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 128, 16, 32, 32)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes=num_classes)
            self.aux2 = InceptionAux(528, num_classes=num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 尺寸变为高*宽 = 1*1
        self.fc = nn.Linear(1024, num_classes)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        pass

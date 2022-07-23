import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os

path = os.getcwd()
print(path)
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在使用的是： ", Device)

train_path = "../../1.MyDataset/Archive/seg_train/seg_train/"
val_path = "../../1.MyDataset/Archive/seg_test/seg_test/"

mytransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = ImageFolder(root=train_path, transform=mytransform)
tran_dataLoader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32, num_workers=7)

val_dataset = ImageFolder(root=val_path, transform=mytransform)
val_dataLoader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=32, num_workers=7)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=1, kernel_size=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=padding)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        x_copy = x  # 保存x
        x = self.conv1(x)
        x = F.relu(self.BN1(x))
        x = self.conv2(x)
        x = F.relu(self.BN2(x))
        x = x + x_copy
        x = F.relu(x)
        return x


resnet_BasicBlock = BasicBlock(224, 224, 1)
print(resnet_BasicBlock)
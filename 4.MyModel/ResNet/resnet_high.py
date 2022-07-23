import torch.nn as nn
import torch.nn.functional as F
import torch


# 实现residualblock
class ResidualBlock(nn.Module):
    def __init__(self, inch, ouch, stride=1, shortcut=None):
        self.expension = 4
        super(ResidualBlock, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(inch, ouch, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(ouch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ouch, ouch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ouch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ouch, self.expension*ouch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.expension*ouch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.right = shortcut
        
    def forward(self, x):
        out = self.operation(x)
        #print("out: ", out.shape)
        residual = x if self.right is None else self.right(x)
        #print("residual: ", residual.shape)
        out += residual
        out = self.relu(out)
        return out
        
class ResNet_High(nn.Module):
    def __init__(self, num_classes=6, each_layernum=[3, 4, 23, 3]):
        super(ResNet_High,self).__init__()
        self.num_classes = num_classes
        # 输入size 3*224*224
        self.pre_operation = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),    # 3*224*224 -> 64*112*112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)   # -> 64*56*56
        )
        self.layer1 = self._make_layer(each_layernum[0], 64, 64, 1)
        self.layer2 = self._make_layer(each_layernum[1], 256, 128, 2)
        self.layer3 = self._make_layer(each_layernum[2], 512, 256, 2)
        self.layer4 = self._make_layer(each_layernum[3], 1024, 512, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=7, padding=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pre_operation(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    def _make_layer(self, block_num, inch, ouch, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inch, ouch*4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(ouch*4)
        )
        layers = []
        layers.append(ResidualBlock(inch, ouch, stride=stride, shortcut=shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(inch=ouch*4, ouch=ouch))
        return nn.Sequential(*layers)
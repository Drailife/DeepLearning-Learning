import torch
import torch.nn as nn
import torch.nn.functional as F

res18 = [2, 2, 2, 2]

class ResidualBlock(nn.Module):
    def __init__(self, inch, ouch, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.shortcut = shortcut
        self.operation = nn.Sequential(
            # stride = 2 输出size为输入的一半
            nn.Conv2d(inch, ouch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ouch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ouch,ouch,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ouch)
        )
    
    def forward(self, x):
        out = self.operation(x)
        residual = x if self.shortcut==None else self.shortcut(x)
        out += residual
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self,num_classes=6,res=[2, 2, 2, 2]):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        # 输入size 3*224*224
        self.pre_operation = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),    # 3*224*224 -> 64*112*112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)   # -> 64*56*56
        )
        self.layer1 = self._make_layer(res[0], 64, 64,stride=1)  # ->64*56*56
        self.layer2 = self._make_layer(res[1], 64, 128, stride=2)  # -> 128*28*28
        self.layer3 = self._make_layer(res[2], 128, 256, 2)  # ->256*14*14
        self.layer4 = self._make_layer(res[2], 256, 512, 2)  # ->512*7*7
        self.conv1 = nn.Conv2d(512, 512, kernel_size=7, stride=2, padding=3)  # ->512*3*3
        self.avgpool = nn.AvgPool2d(kernel_size=3)  # ->512*1*1
        self.fc = nn.Linear(512,num_classes) 
        
        
    def _make_layer(self, num_block, inch, ouch, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inch, ouch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(ouch)
        )
        layers = []
        layers.append(ResidualBlock(inch,ouch,stride=stride, shortcut=shortcut))
        
        for i in range(1, num_block):
            layers.append(ResidualBlock(ouch,ouch))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.pre_operation(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        #x = torch.flatten(x,start_dim=1) # 512
        x = torch.argmax(x, dim=1)
        x = self.fc(x)
        return x

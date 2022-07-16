import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# 定义自己的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # -1 代表自动计算 原来为64*1*28*28 现在为64*784
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        return x


model = MyModel()
model.load_state_dict(torch.load('myMinist.pth'))

img = Image.open('../2.MyDataSet/MyNum/4.png')
img = img.convert('L')  # 转灰度图

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
img = transform(img)

img = torch.reshape(img, (1, 1, 28, 28))

model.eval()
with torch.no_grad():
    output = model(img)
print(output)
output_softmax = F.softmax(output,dim=1)
print(output_softmax)
print(output.argmax(dim=1))

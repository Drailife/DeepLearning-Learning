import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

myimg = Image.open('pic.png')  # (W H)
# print(myimg.size)
myimg_array = np.asarray(myimg)  # ( N H W)
# print(myimg_array)
myimg_array = np.swapaxes(myimg_array, 0, 1)  # 交换H轴 和W轴 相当于图片的高和宽互换 图片旋转
# print(myimg_array.shape)
plt.figure(figsize=(5, 4))
plt.imshow(myimg_array)
plt.axis('off')  # 关闭坐标显示
plt.show()

test = torch.ones(1, 1, 3, 3)
print(test)
myconv2 = torch.nn.Conv2d(in_channels=1, out_channels=1,kernel_size=(2,1) ,stride=1, padding=0)
# test = myconv2(test)
mypool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
print(myconv2)
print(mypool)
print(mypool(test))
print(myconv2(test))

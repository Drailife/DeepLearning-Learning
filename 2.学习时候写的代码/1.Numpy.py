import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

#
# a = np.array([1, 2, 3, 4])
# # b = np.array([2, 3, 4, 5])
# # c = a + b
# # print(c)
#
# a = np.array([1, 2, 3, 4])
# print(a + 1)
# # 二维数组和一维数组相加
# b = np.array([[1, 2, 3, 4], [1, 2, 3, 5]])
# print(b + a)
#
# # 创建ndarray数组
# a = [1, 2, 3, 6]
# a = np.array(a)
# print(a)
#
# b = np.arange(start=0, stop=10, step=1)
# print(b)
# # 创建全0的ndarray
# a = np.zeros([3, 3])  # array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
# # 创建全为一的ndarray
# b = np.ones([3, 3])  # array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
#
# c = np.random.randn(3, 3)
# print(c)
# print(c.shape, c.dtype, c.ndim)
#
# d = np.arange(10)
# print(d)
# d = np.arange(0, 10)
# d = d.reshape(5, 2)
# print(d)

img = Image.open("./pic.png")
print(img)

e = np.arange(10)
print(e)
e1 = e[::]
print(e[:])
print(e[::-1])
img_np = np.array(img)
print(img_np.shape)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(img_np)
ax2.imshow(img_np[::-1, :, :])
ax3.imshow(img_np[:, ::-1, :])
ax4.imshow(img_np[:, ::, ::-1])
# 保存图片


# plt.show()
a = np.array([1, 2, 3])
print(a, a.dtype, sep='\t')
a_tor = torch.from_numpy(a.astype(np.float32))
print(a_tor, a_tor.dtype, sep='\t')

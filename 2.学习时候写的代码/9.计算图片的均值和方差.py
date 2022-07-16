import numpy as np
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import tqdm

path = '..//3.MyDataSet//Archive//seg_train'
transform = transforms.ToTensor()
data_set = torchvision.datasets.ImageFolder(root=path, transform=transform)
num_imgs = len(data_set)  # 获取数据集的图片数量
means, stds = [0, 0, 0], [0, 0, 0]  # 初始化均值和方差
data_bar = tqdm.tqdm(data_set)
for i, (imgs, lables) in enumerate(data_bar):
    for channel in range(3):  # 遍 历图片的RGB三通道
        # 计算每一个通道的均值和标准差
        means[channel] += imgs[channel, :, :].mean()
        stds[channel] += imgs[channel, :, :].std()
    data_bar.set_description(f"[{i}/{num_imgs}]")
mean = np.array(means) / num_imgs
std = np.array(stds) / num_imgs  # 要使数据集归一化，均值和方差需除以总图片数量
print(mean, std)  # 打印出结果

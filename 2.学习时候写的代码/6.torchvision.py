import torchvision
import torch
import torch.utils.data as Datas
# 下载数据集
train_data = torchvision.datasets.FashionMNIST(root='.\ImageDatas\FashionMNIST',
                                               train=True,
                                               download=True,
                                               # 将图像数据由【H, W, C】转换为【C, H, W】
                                               transform=torchvision.transforms.ToTensor()
                                               )
train_loader = Datas.DataLoader(dataset=train_data,
                                batch_size=16,  # 批处理样本大小
                                shuffle=True,   # 每次迭代前打乱数据
                                num_workers=2   # 使用两个进程
                                )

print("trian_loader's size: ", len(train_loader))
torchvision.datasets.ImageFolder()

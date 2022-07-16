import sklearn
from sklearn import datasets
import torch
import numpy as np
import torch.utils.data as Data

boston_x, boston_y = datasets.load_boston(return_X_y=True)
print(boston_x.shape)
print(boston_y.shape)
# 训练集x y 转换为张量
# pytorch 使用的数据为torch的32位浮点型的张量
train_xt = torch.from_numpy(boston_x.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))

BATCH_SIZE = 5
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
print(torch_dataset)
print(loader)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training
            print("step:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    show_batch()

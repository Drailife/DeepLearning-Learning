import time

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset


# 定义自己的Dataset
class MyDataset(Dataset):
    def __init__(self, datasname):
        super(MyDataset, self).__init__()
        xy_datas = np.loadtxt(datasname, delimiter=',', dtype=np.float32)  # 读取csv中的文本数据
        self.length = xy_datas.shape[0]  # 得到数据的数量
        # print(xy_datas)
        # print(xy_datas.shape)   # (759, 9)
        self.x_datas = xy_datas[:, :-1]  # 表示截取除去最后一列的所有数据，若为[: , :-2]则为获取除去最后两列的所有数据 [行 ， 列]
        self.y_datas = xy_datas[:, [-1]]  # 表示获取最后一列的所有数据 [: , [n]] 表示获取第n列的数据

    def __getitem__(self, item):
        return self.x_datas[item], self.y_datas[item]

    def __len__(self):
        return self.length


# 定义自己的线性回归模型
class MyModle(nn.Module):
    def __init__(self):
        super(MyModle, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    # 重写前向传播函数
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


dataset = MyDataset('..\\MyDatas\\diabetes.csv.gz')
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
my_modle = MyModle()  # 创建一个模型的实例化对象
criterion = nn.BCELoss()  # 二分类问题，选用Bceloss 二分类交叉熵损失函数计算损失值
optimizer = torch.optim.Adam(params=my_modle.parameters(), lr=0.01)  # 选用随机梯度下降算法定义优化器

train_epoch = 10000
my_modle.train()
if __name__ == '__main__':
    for epoch in range(train_epoch):
        print('-----', epoch)
        for i, datas in enumerate(dataloader):
            # print(datas)
            # 1. Prepared datas
            data, lables = datas
            # 2. Forward
            lable_pred = my_modle(data)
            loss = criterion(lables, lable_pred)  # 计算损失值
            print('\t', loss.item())
            # 3. Backward
            loss.backward()  # 反向传播
            optimizer.zero_grad()  # 梯度清零
            # 4. Update
            optimizer.step()  # 梯度更新

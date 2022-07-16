import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('你正在使用的为： ', Device)
# 加载数据集
transform = transforms.Compose([transforms.ToTensor(),  # 将数据变成[0,1] 直接除以255得到的
                                transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                                # 将数据规划到【-1， 1】 计算方法  x = (x-mean)/std
                                # mean 和 std是前人经过计算得到的
                                ])
trian_dataset = torchvision.datasets.MNIST(root='.\MyDatas\Train_MNIST手写字体',
                                           train=True,
                                           download=True,
                                           transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=trian_dataset,
                                               batch_size=64,
                                               shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='.\MyDatas\Test_MNIST手写字体',
                                          train=False,
                                          download=True,
                                          transform=transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=True)


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
model = model.to(Device)
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
criterion = criterion.to(Device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 随机梯度下降

train_epochs = 20
model.train()
loss_plot = []
acc_plot = []

for epoch in range(train_epochs):
    train_dataloader_bar = tqdm.tqdm(train_dataloader, colour='red')
    each_epoch_loss = 0
    for batch_idx, (datas, lables) in enumerate(train_dataloader_bar):
        datas = datas.to(Device)
        lables = lables.to(Device)
        lables_pre = model(datas)
        each_ite_loss = criterion(lables_pre, lables)
        each_epoch_loss += each_ite_loss.item()
        optimizer.zero_grad()  # 梯度清零
        each_ite_loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        train_dataloader_bar.set_description(
            'train epoch[{}/{}] loss={:.3f}'.format(epoch + 1, train_epochs, each_ite_loss))  # 添加描述
    loss_plot.append(each_epoch_loss)
    # print('epoch: ', epoch, 'loss: ', each_epoch_loss)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(Device)
            labels = labels.to(Device)
            outputs = model(images)
            predicted = torch.max(outputs, dim=1)[1]
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()  # 计算正确率
        acc_plot.append(100 * correct / total)
        # print('[epoch %d] Accuracy on test set: %d %%' % (epoch, 100 * correct / total))
        # 避免和print打印冲突
        tqdm.tqdm.write('[epoch %d] Accuracy on test set: %d %%' % (epoch, 100 * correct / total))
        if 100 * correct / total >= 97:
            torch.save(model.state_dict(),'myMinist.pth')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_plot)
plt.subplot(1, 2, 2)
plt.plot(acc_plot)
plt.show()

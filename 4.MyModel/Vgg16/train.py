import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from VGG16 import Vgg16

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('你正在使用的为： ', Device)

traindata_path = '..//..//3.MyDataSet//Archive//seg_train'
valdata_path = '..//..//3.MyDataSet//Archive//seg_test'
my_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.43017164, 0.4574746, 0.4538466],
                                                        [0.23553437, 0.23454581, 0.24294421])])
train_dataset = ImageFolder(root=traindata_path, transform=my_transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder(root=valdata_path, transform=my_transform)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

myvgg16 = Vgg16(num_classes=6)
myvgg16.to(Device)

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
criterion = criterion.to(Device)
optimizer = torch.optim.SGD(myvgg16.parameters(), lr=0.01, momentum=0.5)  # 随机梯度下降

epochs = 10
myvgg16.train()
train_epochloss = []
val_acc = []

for epoch in range(epochs):
    epoch_loss = 0
    train_bar = tqdm.tqdm(train_dataloader, colour='red')
    myvgg16.train()
    for i, datas in enumerate(train_bar):
        img, lables = datas
        img = img.to(Device)
        lables = lables.to(Device)
        lables_pred = myvgg16(img)
        iter_loss = criterion(input=lables_pred, target=lables)
        optimizer.zero_grad()
        iter_loss.backward()
        optimizer.step()
        train_bar.set_description('train epoch[{}/{}] loss={:.3f}'.format(epoch + 1, epochs, iter_loss.item()))
        epoch_loss += iter_loss.item()
    train_epochloss.append(epoch_loss)

    myvgg16.eval()
    correct = 0
    total = len(val_dataloader)*32
    tqdm.tqdm.write("数据总量为：{}".format(total))
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images = images.to(Device)
            labels = labels.to(Device)
            outputs = myvgg16(images)
            predicted = torch.max(outputs, dim=1)[1]
            correct += (predicted == labels).sum().item()  # 计算正确率
        val_acc.append(100 * correct / total)
        # print('[epoch %d] Accuracy on test set: %d %%' % (epoch, 100 * correct / total))
        # 避免和print打印冲突
        tqdm.tqdm.write('[epoch %d] Accuracy on test set: %.4f %%' % (epoch + 1, 100 * correct / total))
        tqdm.tqdm.write("正确个数: [{}/{}]".format(correct, total))
        torch.save(myvgg16.state_dict(),'./Vgg16_classfy_Archive.pth')

fig = plt.figure(figsize=(4, 4))
plt.subplot(1, 2, 1)
plt.plot(train_epochloss)
plt.subplot(1, 2, 2)
plt.plot(val_acc)
plt.show()

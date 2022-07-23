import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from resnet_high import ResNet_High

run_device = 'Windows'  # 设置了num_work在俩个设备上运行不一样
path = os.getcwd()
print(path)
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在使用的是： ", Device)

# 天翼云服务器
# train_path = '..//Drailife_MyDataset/Archive/seg_train/seg_train'
# val_path = '..//Drailife_MyDataset/Archive/seg_test/seg_test'
# AutoDL服务器
# traindata_path = '..//Drailife_MyDataset/Archive/seg_train/seg_train'
# valdata_path = '..//Drailife_MyDataset/Archive/seg_test/seg_test'

# 本地端
train_path = '../../3.MyDataSet/DataSet/Archive/seg_train/seg_train'
val_path = '../../3.MyDataSet/DataSet/Archive/seg_test/seg_test'
batch_size = 32
num_workers = 8 if run_device == 'Linux' else 0
mytransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = ImageFolder(root=train_path, transform=mytransform)
train_dataLoader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

val_dataset = ImageFolder(root=val_path, transform=mytransform)
val_dataLoader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

resnet = ResNet_High(num_classes=6, each_layernum=[3, 4, 23, 3])
print(resnet)
resnet = resnet.to(Device)
cer = nn.CrossEntropyLoss()
cer = cer.to(Device)
opti = torch.optim.SGD(resnet.parameters(), lr=0.001)

resnet.train()
epochs = 100
total_train_loss = []
total_val_loss = []
total_train_acc = []
total_val_acc = []
total_datas_num = len(train_dataset)

for epoch in range(epochs):
    start_time = time.time()
    each_epoch_curr = 0
    each_epoch_trainloss_sum = 0
    data_bar = tqdm.tqdm(train_dataLoader)
    for i, (img, lables) in enumerate(data_bar, 0):
        img, lables = img.to(Device), lables.to(Device)
        pre_lables = resnet(img)
        iter_loss = cer(pre_lables, lables)
        true_nums = (torch.argmax(pre_lables, dim=1) == lables).sum().item()  # dim=0 表示列最大  dim = 1表示行最大
        each_epoch_curr += true_nums
        data_bar.set_description("train epoch:[{}/{}] loss = {:.3f}".format(epoch + 1, epochs, iter_loss.item()))
        each_epoch_trainloss_sum += iter_loss.item()

        opti.zero_grad()
        iter_loss.backward()
        opti.step()
    total_train_loss.append(each_epoch_trainloss_sum)
    data_bar.close()
    tqdm.tqdm.write(
        'Train: Average loss of each batch: {:.3}'.format(each_epoch_trainloss_sum / total_datas_num * batch_size))
    tqdm.tqdm.write("Train: Total Correct number of all datas: {}/{}, Accuracy rate: {:.4}%".format(each_epoch_curr,
                                                                                                    total_datas_num,
                                                                                                    each_epoch_curr / total_datas_num * 100))

    total_train_acc.append(each_epoch_curr / total_datas_num * 100)  # 正确率
    # 验证
    resnet.eval()
    with torch.no_grad():
        each_epoch_curr = 0
        each_epoch_valloss_sum = 0
        for (img, lables) in val_dataLoader:
            img, lables = img.to(Device), lables.to(Device)
            pre_lables = resnet(img)
            iter_loss = cer(pre_lables, lables)
            true_nums = (torch.argmax(pre_lables, dim=1) == lables).sum().item()  # dim=0 表示列最大  dim = 1表示行最大
            each_epoch_curr += true_nums
            each_epoch_valloss_sum += iter_loss.item()
    total_val_loss.append(each_epoch_valloss_sum)
    tqdm.tqdm.write('Val: Average loss of each batch: {:.3}'.format(each_epoch_valloss_sum / len(val_dataLoader)))
    tqdm.tqdm.write("Val: Total Correct number of all datas: {}/{}, Accuracy rate: {:.4}%".format(each_epoch_curr,
                                                                                                  len(val_dataset),
                                                                                                  each_epoch_curr / len(
                                                                                                      val_dataset) * 100))
    total_val_acc.append(each_epoch_curr / len(val_dataset) * 100)
    end_time = time.time()
    tqdm.tqdm.write("Each Epoch Time: {:.4f}".format(end_time - start_time))

plt.figure()
plt.subplot(1, 4, 1)
plt.title("Train_Loss")
plt.plot(total_train_loss)
plt.subplot(1, 4, 2)
plt.title("Train_ACC")
plt.plot(total_train_acc)
plt.subplot(1, 4, 3)
plt.title("Val_Loss")
plt.plot(total_val_loss)
plt.subplot(1, 4, 4)
plt.title("Val_Acc")
plt.plot(total_val_acc)
plt.show()

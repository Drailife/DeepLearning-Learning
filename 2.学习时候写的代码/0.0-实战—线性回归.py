import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# 数据
x_data = torch.Tensor([[1.], [2.], [3.], [4.]])
y_data = torch.Tensor([[2.0], [4.0], [6.0], [8.0]])


# 线性回归网络模型
class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


MyModele = LinearModule()  # 模型的实例化对象
criterion = nn.SmoothL1Loss(size_average=False)  # 损失器
optimizer = torch.optim.ASGD(MyModele.parameters(), lr=0.001)
loss_list = []
for epoch in range(999):
    y_pred = MyModele(x_data)  # 计算预测值
    loss = criterion(y_data, y_pred)  # 计算损失值
    loss_list.append(loss.item())
    print(epoch, loss.item())

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 梯度更新
    # print('w = ', MyModele.linear.weight.item())
    # print('b = ', MyModele.linear.bias.item())

print('w = ', MyModele.linear.weight.item())
print('b = ', MyModele.linear.bias.item())

x_test = torch.Tensor([[20.0]])
y_test = MyModele(x_test)
print('y_pred = ', y_test)
plt.plot(loss_list)
plt.show()
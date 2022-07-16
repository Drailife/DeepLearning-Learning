import torch
import torch.nn as nn

loss1 = nn.MSELoss(reduction='sum')
loss2 = nn.MSELoss(reduction='mean')
inputs = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float32)  # tensor([1., 2., 3.], requires_grad=True)
target = torch.tensor([0, 2, 4], dtype=torch.float32)  # tensor([1., 2., 4.])
# 若为 sum : 计算为 (1 -0)**2 + (2 - 2)**2 + (3-4)**2 = 2.0
output1 = loss1(inputs, target)  # tensor(2., grad_fn=<MseLossBackward0>)

# 若为 mean : 计算为 ((1 -0)**2 + (2 - 2)**2 + (3-4)**2) / 3 = 0.666667
output2 = loss2(inputs, target)  # tensor(0.6667, grad_fn=<MseLossBackward0>)

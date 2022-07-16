import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
x = torch.linspace(-10, 10, 100)
print(x.data)
print(x)
myrelu = nn.ReLU()
yrelu = myrelu(x)
print(yrelu)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(x.data.numpy(),yrelu.data.numpy())
plt.grid()
plt.show()
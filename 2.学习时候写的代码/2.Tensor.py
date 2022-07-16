import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cup"
print(DEVICE)

c = torch.tensor(range(15))
print(c)
c = c.reshape(1,3,5)
d = torch.linspace(start=0, end=10, steps=4)

print(d)
a = torch.tensor([1.], requires_grad=True)
print(a)
s = a**2
print(s)
# torch.autograd.backward(s)

print(a.grad)
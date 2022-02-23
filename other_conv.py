import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


class DeptwiseConv(nn.Module):
    def __init__(self, nin, kernel_size):
        super(DeptwiseConv, self).__init__()
        self.deptwise = nn.Conv2d(nin, nin, kernel_size, groups = nin)
        
class PointwiseConv(nn.Module):
    def __init__(self, nin):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv2d(nin, 1, 1)

indata = datasets.FashionMNIST(root='data', train = True, download=True, transform=ToTensor())
indata_loader = DataLoader(indata, batch_size=1)
train_features, train_labels = next(iter(indata_loader))

print(train_features)

# DC = DeptwiseConv(1, 5)
# trans = nn.ConvTranspose2d(1, 1, 5)

# x = DC(train_features)
# y = trans(x)


# after_conv = DC(Fin)
# after_trans = trans(after_conv)


# figure = plt.figure(figsize=(8,4))
# cols, rows = 3, 3

# img, label = indata[sample_index]
# figure.add_subplot(rows, cols, 1)

# plt.axis("off")
# plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
















# channel_size = 1
# input_data = torch.randn(1, channel_size, 5, 5)
# dc = DeptwiseConv(channel_size, 3)
# trans = nn.ConvTranspose2d(channel_size, channel_size, 3)

# x = dc.deptwise(input_data)
# y = trans(x)
# print(input_data,"\n",x, "\n", y)


# dc = DeptwiseConv(channel_size, 3)
# pc = PointwiseConv(channel_size)

# x = dc.deptwise(input_data)
# y = pc.pointwise(x)
# print(input_data, input_data.size())
# print(y, y.size())

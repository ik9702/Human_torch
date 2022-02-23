import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class DeptwiseConv(nn.Module):
    def __init__(self, nin, kernel_size,stri, pad):
        super(DeptwiseConv, self).__init__()
        self.deptwise = nn.Conv2d(nin, nin, kernel_size,stride=stri, padding=pad, groups=nin)
        
class PointwiseConv(nn.Module):
    def __init__(self, nin):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv2d(nin, 1, 1)

indata = datasets.FashionMNIST(root='data', train = False, download=True, transform=ToTensor())
indata_loader = DataLoader(indata, batch_size=1, shuffle=True)
train_features, train_labels = next(iter(indata_loader))

kernel_size = 8
pad = 5
stri = 4
DC = DeptwiseConv(1, kernel_size,stri, pad)
trans = nn.ConvTranspose2d(1, 1, kernel_size,stride = stri, padding = pad)

x = DC.deptwise(train_features)
y = trans(x)

x_t = torch.tensor(x)
y_t = torch.tensor(y)

img1 = train_features[0].squeeze()
plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)
plt.imshow(img1, cmap="gray")
plt.title(train_features.size())

img2 = x_t[0].squeeze()
plt.subplot(1, 3, 2)
plt.imshow(img2, cmap="gray")
plt.title(x_t.size())

img3 = y_t[0].squeeze()
plt.subplot(1, 3, 3)
plt.title(y_t.size())
plt.imshow(img3, cmap="gray")
plt.show()




 












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

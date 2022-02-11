import torch
from torch import nn
import numpy as np


# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)
# print(type(data), type(x_data))


# np_data = np.array(data)
# x_np = torch.from_numpy(np_data)
# print(type(x_np))



# device = 'cuda' if torch.cuda.is_available else 'cup'
# print(f"using {device} device")

# tensor = torch.ones(4,4)
# print('First row', tensor[0])
# print('First column:', tensor[:,0])
# print('Last column:', tensor[...,-1])
# tensor[:,1] = 0
# print(tensor)

# t1 = torch.cat([tensor, tensor, tensor], dim=0)
# t1 = torch.cat([t1, t1, t1], dim=1)
# print(t1)


tensor = torch.ones(3,4)
tensor[:,0] = 0

# print(tensor)
# print(tensor.T)

# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)
# y3 = torch.rand(0)
# torch.matmul(tensor, tensor.T, out=y3)

# print(y1,"\n", y2,"\n", y3)

print(tensor)
# tensor.T_()
print(tensor.T)
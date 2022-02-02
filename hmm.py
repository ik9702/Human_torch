import numpy as np
import torch
import numpy as np


print("hmm")

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

shape = (2,2,)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(x_ones, x_rand)
print(x_rand.shape)
print(x_ones.device)

if torch.cuda.is_available():
    x_ones = x_ones.to('cuda')

print(x_ones.device)
import torch

data = [1,2,3,4,5]
m = torch.tensor(data)
print(m)

print(m.argmax(0))
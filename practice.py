import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


device = "cuda" if torch.cuda.is_available else "cpu"
print(f"device: {device}")

batch_size = 64
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor(),)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor(),)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)



class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
      )
    
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits


model = NeuralNetwork().to(device)
print(model)
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X,y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      print(f"loss: {loss:>6f} [{batch*size:>d}/{size:>d}]")

def test(dataloader, model, loss_fn):
  model.eval()
  test_loss = 0
  correct = 0
  size = len(dataloader.dataset)
  num = len(dataloader)
  with torch.no_grad:
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1)==y).type(torch.float).sum().item()
  test_loss /= num
  correct /= size
  print(f"test error: \naccuracy: {correct}, average loss: {test_loss}")
  
epochs = 5
for t in range(1, epochs+1):
  print(f"epoch {epochs}\n------------------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  test(test_dataloader, model, loss_fn)
  
print("done!")
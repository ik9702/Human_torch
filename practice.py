from sympy import linear_eq_to_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')



training_data = datasets.FashionMNIST(root='data', 
                                      train=True, 
                                      download=True, 
                                      transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', 
                                  train=True, 
                                  download=True, 
                                  transform=ToTensor())

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # =>NeuralNetwork.__init__(self)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28**2, 512),
                                          nn.ReLU(),
                                          nn.Linear(512,512),
                                          nn.ReLU(),
                                          nn.Linear(512, 10))
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            print(f"loss: {loss:>8f} [{batch*batch_size:>5d}/{batch_size*len(dataloader.dataset):>5d}]")
            
            
            
            
def test(dataloader, model, loss_fn):
    model.eval()
    correct = 0
    test_loss = 0
     
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().type(torch.float).item()
            test_loss += loss_fn(pred, y)
        correct /= len(dataloader.dataset)
        avg_loss = test_loss / len(dataloader)  
    print(f"Test error: \n Accuracy: {100*correct/len(dataloader.dataset):>0.1f} | Avg loss: {avg_loss:>8f}")




epoches = 5

for t in range(epoches):
    print(f"epoch: {t+1} \n------------------------------------------------")
    train(test_dataloader, model, loss_fn, optimizer)
    test(train_dataloader, model, loss_fn)
    
print('Done!')

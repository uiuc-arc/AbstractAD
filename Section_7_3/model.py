import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.num_layers = layers
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)

        if layers >= 4:
            self.fc3 = nn.Linear(100, 100)
        if layers == 5:
            self.fc4 = nn.Linear(100, 100)
        
        self.fc_final = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.softplus(self.fc1(x), threshold=100)
        x = F.softplus(self.fc2(x), threshold=100)

        if self.num_layers >= 4:
            x = F.softplus(self.fc3(x), threshold=100)
        if self.num_layers == 5:
            x = F.softplus(self.fc4(x), threshold=100)

        x = self.fc_final(x)
        return x

def FCNBig():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024),
        nn.Softplus(threshold=100),
        nn.Linear(1024, 1024),
        nn.Softplus(threshold=100),
        nn.Linear(1024, 1024),
        nn.Softplus(threshold=100),
        nn.Linear(1024, 1024),
        nn.Softplus(threshold=100),
        nn.Linear(1024, 10)
    )

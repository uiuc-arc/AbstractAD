import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from timeit import default_timer as timer
import os

import argparse
parser = argparse.ArgumentParser(description='Train MNIST Network')
parser.add_argument('--network', choices=['3layer', '4layer', '5layer', 'big'], help='neural network architecture')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load dataset
batch_size = 256
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./MNIST_Data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4, pin_memory=True)

# initialize network
from model import FCN, FCNBig

if args.network == '3layer':
    net = FCN(3).to(device)
elif args.network == '4layer':
    net = FCN(4).to(device)
elif args.network == '5layer':
    net = FCN(5).to(device)
else:
    net = FCNBig().to(device)

summary(net, (1, 784))

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.75)

# start training
num_epochs = 100

print(f'Start training network')
total_time = 0

for epoch in range(num_epochs):
    start = timer()
    running_loss = 0.0

    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    end = timer()
    t = end - start
    total_time += t
    print(f'Epoch {epoch+1} in {round(t, 5)}s -- Train Loss: {round(running_loss/len(trainset), 3)}')
    scheduler.step()

os.system('mkdir -p trained')
torch.save(net.state_dict(), f'trained/model_{args.network}.pth')

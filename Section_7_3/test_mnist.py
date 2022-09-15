import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser(description='Train MNIST Network')
parser.add_argument('--network', choices=['3layer', '4layer', '5layer', 'big'], help='neural network architecture')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 1024
test_transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root='./MNIST_Data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4, pin_memory=True)

from model import FCN, FCNBig
if args.network == '3layer':
    net = FCN(3).to(device)
elif args.network == '4layer':
    net = FCN(4).to(device)
elif args.network == '5layer':
    net = FCN(5).to(device)
else:
    net = FCNBig().to(device)
net.load_state_dict(torch.load(f'trained/model_{args.network}.pth', map_location=device))

correct = 0
total = 0
batch_n = 0
correct_indices = []
with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct_indices += list(batch_n * batch_size + torch.nonzero((predicted == labels)).flatten())
        batch_n += 1

print('Test Accuracy: %f %%' % (100 * correct / total))
torch.save(correct_indices, f'trained/indices_{args.network}.pth')

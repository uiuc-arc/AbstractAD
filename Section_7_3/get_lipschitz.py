import sys
sys.path.insert(0, '../src')

from SimpleZono import *
from Duals import *

import os
from timeit import default_timer as timer
import torch
import torchvision
import torchvision.transforms as transforms
torch.set_default_dtype(torch.float64)

import argparse
parser = argparse.ArgumentParser(description='Get haze Lipschitz constant of MNIST Network')
parser.add_argument('--network', choices=['3layer', '4layer', '5layer', 'big'], help='neural network architecture')
args = parser.parse_args()

num_images_to_test = 1000

test_transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root='./MNIST_Data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

from model import FCN, FCNBig

network = args.network
print(f'===== {network} Network =====')

if network == '3layer':
    net = FCN(3)
    layers = 3
elif network == '4layer':
    net = FCN(4)
    layers = 4
elif network == '5layer':
    net = FCN(5)
    layers = 5
else:
    net = FCNBig()
net.load_state_dict(torch.load(f'trained/model_{network}.pth', map_location='cpu'))
net.eval()

if 'layer' not in network:  # define forward functions for the "big" network
    def forward_zono(x):
        x = SmoothReluDualZonotope(AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
        x = SmoothReluDualZonotope(AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
        x = SmoothReluDualZonotope(AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
        x = SmoothReluDualZonotope(AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
        x = AffineDualZonotope(x, net[9].weight.T) + net[9].bias
        return x

    def forward_interval(x):
        x = SmoothRelu_di(x @ abstract_di(net[1].weight.T) + abstract_di(net[1].bias))
        x = SmoothRelu_di(x @ abstract_di(net[3].weight.T) + abstract_di(net[3].bias))
        x = SmoothRelu_di(x @ abstract_di(net[5].weight.T) + abstract_di(net[5].bias))
        x = SmoothRelu_di(x @ abstract_di(net[7].weight.T) + abstract_di(net[7].bias))
        x = x @ abstract_di(net[9].weight.T) + abstract_di(net[9].bias)
        return x
else:  # define forward functions for the "3/4/5-layer" networks
    def forward_zono(x):
        x = SmoothReluDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias)
        x = SmoothReluDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias)
        if layers >= 4:
            x = SmoothReluDualZonotope(AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias)
        if layers == 5:
            x = SmoothReluDualZonotope(AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias)
        x = AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias
        return x

    def forward_interval(x):
        x = SmoothRelu_di(x @ abstract_di(net.fc1.weight.T) + abstract_di(net.fc1.bias))
        x = SmoothRelu_di(x @ abstract_di(net.fc2.weight.T) + abstract_di(net.fc2.bias))
        if layers >= 4:
            x = SmoothRelu_di(x @ abstract_di(net.fc3.weight.T) + abstract_di(net.fc3.bias))
        if layers == 5:
            x = SmoothRelu_di(x @ abstract_di(net.fc4.weight.T) + abstract_di(net.fc4.bias))
        x = x @ abstract_di(net.fc_final.weight.T) + abstract_di(net.fc_final.bias)
        return x

correct_indices = torch.load(f'trained/indices_{network}.pth')

lc_zonos = []
lc_intervals = []
time_zonos = []
time_intervals = []

for epsilon in [10**(-k/4) * 2 for k in range(2, 18)]:
    print(f'Running epsilon={epsilon}')
    lc_zono_total = 0
    lc_interval_total = 0
    time_zono_total = 0
    time_interval_total = 0

    img_index = 0
    correct_images = 0

    with torch.no_grad():
        for (image, _) in testloader:
            if img_index in correct_indices:
                img_f = image.flatten()

                interval_eps = abstract_di(torch.tensor([epsilon / 2]), epsilon / 2)
                interval_eps.e1_l[0] = 1; interval_eps.e1_u[0] = 1

                # zonotope
                start = timer()
                zono_eps = HyperDualIntervalToDualZonotope(interval_eps)
                hazed_zono = (torch.tensor([1]) - zono_eps) * img_f + zono_eps
                output = forward_zono(hazed_zono)
                end = timer()
                zono_time = end - start
                lc_zono = torch.max(torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                
                # interval
                start = timer()
                hazed_int = (1 - interval_eps) * img_f + interval_eps
                output = forward_interval(hazed_int)
                end = timer()
                interval_time = end - start
                lc_interval = torch.max(torch.maximum(torch.abs(output.e1_l), torch.abs(output.e1_u))).item()
                
                time_zono_total += zono_time
                time_interval_total += interval_time
                lc_zono_total += lc_zono
                lc_interval_total += lc_interval
                correct_images += 1

            if correct_images == num_images_to_test:
                break

            img_index += 1
    
    lc_zonos.append(lc_zono_total / num_images_to_test)
    lc_intervals.append(lc_interval_total / num_images_to_test)
    time_zonos.append(time_zono_total / num_images_to_test)
    time_intervals.append(time_interval_total / num_images_to_test)

os.system('mkdir -p results')
torch.save(lc_zonos, f'results/lc_zonos_{network}.pth')
torch.save(lc_intervals, f'results/lc_intervals_{network}.pth')
torch.save(time_zonos, f'results/time_zonos_{network}.pth')
torch.save(time_intervals, f'results/time_intervals_{network}.pth')

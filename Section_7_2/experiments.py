import sys
sys.path.insert(0, '../src')

from HyperDuals import *
from Duals import *
from HyperDualZono import *
from SimpleZono import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
from timeit import default_timer as timer
sns.set_style('darkgrid')

num_interactions = 5

import argparse
parser = argparse.ArgumentParser(description='Robust Interpretations Experiment')
parser.add_argument('--seed', type=int)  # seed for RNGs
parser.add_argument('--network-file')    # path from which to load pretrained network
args = parser.parse_args()

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class Polynomial(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        out = F.softplus(self.fc1(x), threshold=100)
        out = F.softplus(self.fc2(out), threshold=100)
        out = F.softplus(self.fc3(out), threshold=100)
        return out
    
    def forward_interval(self, x):
        x = x @ abstract_di(self.fc1.weight.t()) + abstract_di(self.fc1.bias)
        x = SmoothRelu_di(x)

        x = x @ abstract_di(self.fc2.weight.t()) + abstract_di(self.fc2.bias)
        x = SmoothRelu_di(x)

        x = x @ abstract_di(self.fc3.weight.t()) + abstract_di(self.fc3.bias)
        x = SmoothRelu_di(x)
        return x
    
    def forward_hd(self, x):
        x = x @ abstract_hdi(self.fc1.weight.t()) + abstract_hdi(self.fc1.bias)
        x = SmoothRelu_hdi(x)

        x = x @ abstract_hdi(self.fc2.weight.t()) + abstract_hdi(self.fc2.bias)
        x = SmoothRelu_hdi(x)

        x = x @ abstract_hdi(self.fc3.weight.t()) + abstract_hdi(self.fc3.bias)
        x = SmoothRelu_hdi(x)
        return x

    def forward_zono(self,x):
        x = AffineDualZonotope(x, self.fc1.weight.t()) + self.fc1.bias
        x = SmoothReluDualZonotope(x)

        x = AffineDualZonotope(x, self.fc2.weight.t()) + self.fc2.bias
        x = SmoothReluDualZonotope(x)

        x = AffineDualZonotope(x, self.fc3.weight.t()) + self.fc3.bias
        x = SmoothReluDualZonotope(x)
        return x

    def forward_zono_hd(self,x):
        x = AffineHyperDualZonotope(x, self.fc1.weight.t()) + self.fc1.bias
        x = SmoothReluHyperDualZonotope(x)

        x = AffineHyperDualZonotope(x, self.fc2.weight.t()) + self.fc2.bias
        x = SmoothReluHyperDualZonotope(x)

        x = AffineHyperDualZonotope(x, self.fc3.weight.t()) + self.fc3.bias
        x = SmoothReluHyperDualZonotope(x)
        return x

def generate_polynomial(inputs, features, P=10, size=10000):
    # generate a random k-degree polynomial with P interactions
    poly = []
    outputs = torch.zeros((size, 1))
    f1f2 = random.sample(list(itertools.combinations(range(features), 2)), P)

    for f1, f2 in f1f2:
        alpha = random.random()
        print(alpha, f1, f2)
        poly.append((alpha, f1, f2))
        outputs += (alpha * inputs[:,f1]  * inputs[:,f2]).unsqueeze(1)
    
    return outputs, poly

class Dataset(torch.utils.data.Dataset):
  def __init__(self, inputs, outputs):
        'Initialization'
        self.inputs = inputs
        self.outputs = outputs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.inputs[index]
        y = self.outputs[index]
        return X, y

def first_nn_der(net, x, features, eps, samples=100):
    ders = np.zeros((features, samples))
    for i in range(samples):
        x_cur = (x-eps)+ torch.rand(x.shape)*2*eps 
        assert ((x-eps) <= x_cur).all() and ((x+eps) >= x_cur).all()
        jac = torch.autograd.functional.jacobian(net, x_cur)
        ders[:, i] = jac.squeeze()
    return ders

def first_nn_hd_der(net, x, features, eps):
    ders = []
    for i in range(features):
        ab = abstract_di(x, eps)
        ab.e1_l[0, i] = 1
        ab.e1_u[0, i] = 1
        out = net.forward_interval(ab)
        ders.append((out.e1_l[0].item(), out.e1_u[0].item()))
    return ders

def first_poly_der(x, poly, xi):
    der = 0
    for alpha, f1, f2 in poly:
        if f1 == xi:
            der += alpha * x[0, f2].item()
        if f2 == xi:
            der += alpha * x[0, f1].item()
    return der

def first_nn_zono_der(net, x, features, eps):
    ders = []
    for i in range(features):
        ab = abstract_di(x, eps)
        ab.e1_l[0, i] = 1
        ab.e1_u[0, i] = 1
        ab = HyperDualIntervalToDualZonotope(ab)
        out = net.forward_zono(ab)
        e1_l, e1_u = ZonotopeToInterval(out.dual)
        ders.append((e1_l[0].item(), e1_u[0].item()))    
    return ders

def plot_first_order(data, bounds, bounds_zono):    
    SMALL_SIZE = 16
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)            # legend fontsize

    labels = [f"$J_{{{i+1}}}$" for i in range(5)]
    colors = ["crimson", "purple", "limegreen", "gold", "green"]

    width = 0.4
    fig, ax = plt.subplots()
    width_ratios = []

    for i, _ in enumerate(labels):
        x = np.ones(data.shape[1])*i + (np.random.rand(data.shape[1])*width-width/2.)
        ax.scatter(x, data[i,:], color=colors[i], s=1)
        # mean = data[i,:].mean()
        (legend_interval,) = ax.plot([i-width/2., i+width/2.],[bounds[i][0],bounds[i][0]], color="k", label="Interval")
        ax.plot([i-width/2., i+width/2.],[bounds[i][1],bounds[i][1]], color="k")
        (legend_zono,) = ax.plot([i-width/2., i+width/2.],[bounds_zono[i][0],bounds_zono[i][0]], color="r", label="Zonotope")
        ax.plot([i-width/2., i+width/2.],[bounds_zono[i][1],bounds_zono[i][1]], color="r")

        zono_width = bounds_zono[i][1] - bounds_zono[i][0]
        interval_width = bounds[i][1] - bounds[i][0]
        width_ratios.append(interval_width / zono_width)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(handles=[legend_interval, legend_zono], loc="upper right")

    print("jacobian: interval width / zonotope width")
    print("\\begin{tabular}{c" + ("|c" * len(labels)) + "}")
    print("Jacobian Component & " + " & ".join(labels) + "\\\\")
    print("\\hline")
    print("Width reduction factor & " + " & ".join(map("{:.3f}".format, width_ratios)) + "\\\\")
    print("\\end{tabular}")

    with open(f'interactions{num_interactions}/first_order.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(width_ratios)

    # plt.show()
    plt.savefig(f'interval_vs_zono_seed{seed}.png', dpi=500, bbox_inches='tight')

def plot_second_order(data, bounds, bounds_zono):
    SMALL_SIZE = 24
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=28)            # legend fontsize

    # Only unique hessian terms -- e.g. 1,5 but not 5,1
    unique_entries = (
        [True,  True,  True,  True,  True,
         False, True,  True,  True,  True,
         False, False, True,  True,  True,
         False, False, False, True,  True,
         False, False, False, False, True
        ]
    )
    data = data[unique_entries]
    bounds = [b for (b, u) in zip(bounds, unique_entries) if u]
    bounds_zono = [b for (b, u) in zip(bounds_zono, unique_entries) if u]
    labels = [f"$H_{{{i+1},{j+1}}}$" for i in range(5) for j in range(5)]
    labels = [l for (l, u) in zip(labels, unique_entries) if u]
    colors = ["crimson", "purple", "limegreen", "gold", "green"]

    width = 0.4
    fig, ax = plt.subplots()
    # fig.set_size_inches(25.5, 10.5)
    fig.set_size_inches(12.5, 8.5)

    width_ratios = []

    for i, _ in enumerate(labels):
        x = np.ones(data.shape[1])*i + (np.random.rand(data.shape[1])*width-width/2.)
        ax.scatter(x, data[i,:], color=colors[i%5], s=1)
        # mean = data[i,:].mean()
        (legend_interval,) = ax.plot([i-width/2., i+width/2.],[bounds[i][0],bounds[i][0]], color="k", label="Interval")
        ax.plot([i-width/2., i+width/2.],[bounds[i][1],bounds[i][1]], color="k")
        (legend_zonotope,) = ax.plot([i-width/2., i+width/2.],[bounds_zono[i][0],bounds_zono[i][0]], color="r", label="Zonotope")
        ax.plot([i-width/2., i+width/2.],[bounds_zono[i][1],bounds_zono[i][1]], color="r")

        zono_width = bounds_zono[i][1] - bounds_zono[i][0]
        interval_width = bounds[i][1] - bounds[i][0]
        width_ratios.append(interval_width / zono_width)

    ax.tick_params(axis="x", labelsize=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(handles=[legend_interval, legend_zonotope], loc="upper right")

    print("hessian: interval width / zonotope width")
    print("\\begin{tabular}{c" + ("|c" * len(labels)) + "}")
    print("Hessian Component & " + " & ".join(labels) + "\\\\")
    print("\\hline")
    print("Width reduction factor & " + " & ".join(map("{:.3f}".format, width_ratios)) + "\\\\")
    print("\\end{tabular}")

    with open(f'interactions{num_interactions}/second_order.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(width_ratios)

    # plt.show()
    plt.savefig(f'interval_vs_zono_2nd_seed{seed}.png', dpi=500, bbox_inches='tight')

def second_nn_der(net, x, features, eps, samples=100):
    hes = np.zeros((features*features, samples))
    for i in range(samples):
        x_cur = (x-eps)+ torch.rand(x.shape)*2*eps
        assert ((x-eps) <= x_cur).all() and ((x+eps) >= x_cur).all()
        hes_torch = torch.autograd.functional.hessian(net, x_cur)
        hes[:, i] = hes_torch.flatten()
    return hes

def second_nn_hd_der(net, x, features, eps):   
    hess = []
    for i in range(features):
        for j in range(features):
            ab = abstract_hdi(x, eps)
            ab.e1_l[0, i] = 1
            ab.e1_u[0, i] = 1
            ab.e2_l[0, j] = 1
            ab.e2_u[0, j] = 1
            out = net.forward_hd(ab)
            hess.append((out.e1e2_l[0].item(), out.e1e2_u[0].item()))
    return hess

def second_nn_zono_der(net, x, features, eps):   
    hess = []
    for i in range(features):
        for j in range(features):
            ab = abstract_hdi(x, eps)
            ab.e1_l[0, i] = 1
            ab.e1_u[0, i] = 1
            ab.e2_l[0, j] = 1
            ab.e2_u[0, j] = 1
            ab = HyperDualIntervalToHyperDualZonotope(ab)
            out = net.forward_zono_hd(ab)
            e1e2_l,e1e2_u = ZonotopeToInterval(out.e1e2)
            hess.append((e1e2_l[0].item(), e1e2_u[0].item()))
    return hess


def main():    
    size = 10000
    features = 5
    eps = 0.01
    net = Polynomial(features)
    inputs = torch.rand((size, features))

    outputs, poly = generate_polynomial(inputs, features, P=num_interactions, size=size)

    train_size = int(0.8*size)
    train_dataset = Dataset(inputs[:train_size], outputs[:train_size])
    test_dataset = Dataset(inputs[train_size:], outputs[train_size:])
    params = {'batch_size': 100}
    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)
    loss_func = torch.nn.MSELoss()
    
    if args.network_file is not None:  # load pretrained
        net_dict, x = torch.load(args.network_file)
        net.load_state_dict(net_dict)
        net.eval()

    else:  # train polynomial network on the spot
        os.system(f'mkdir -p interactions{num_interactions}/networks')
        
        epochs = 100
        optimizer = torch.optim.SGD(net.parameters(), lr=0.3)

        for _ in range(epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                prediction = net(X)
                loss = loss_func(prediction, y)
                loss.backward()
                optimizer.step()

    # test
    num_batches = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            prediction = net(X)
            test_loss += loss_func(prediction, y).item()
            num_batches += 1
    print(f"Test loss: {test_loss / num_batches}")

    # coordinate at which to evaluate derivatives
    if args.network_file is None:
        x = torch.rand((1, features))
        torch.save([net.state_dict(), x], f'interactions{num_interactions}/networks/net{seed}.pt')

    ## First derivative analysis
    print("Actual first derivative of the polynomial:")
    for i in range(features):
        print(first_poly_der(x, poly, i))

    print("First derivative of the neural network:")
    data = first_nn_der(net, x, features, eps)
    print(data)

    print("First derivative from HD (Intervals):")
    start = timer()
    bounds = first_nn_hd_der(net, x, features, eps)
    end = timer()
    time_first_interval = end - start
    print(f"TIME FIRST INTERVAL: {time_first_interval}")
    print(bounds)

    print("First derivative from Zonos:")
    start = timer()
    bounds_zono = first_nn_zono_der(net, x, features, eps)
    end = timer()
    time_first_zono = end - start
    print(f"TIME FIRST ZONO: {time_first_zono}")
    print(bounds_zono)

    with open(f'interactions{num_interactions}/time_first.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_first_zono, time_first_interval])

    plot_first_order(data, bounds, bounds_zono)
    
    ## Second order analysis
    print("Second derivative of the neural network:")
    hes_data = second_nn_der(net, x, features, eps)
    print(hes_data)

    print("Second derivative from HD (Intervals):")
    start = timer()
    hdi_hess = second_nn_hd_der(net, x, features, eps)
    end = timer()
    time_second_interval = end - start
    print(f"TIME SECOND INTERVAL: {time_second_interval}")
    print(hdi_hess)

    print("Second derivative from Zonotopes:")
    start = timer()
    zono_hess = second_nn_zono_der(net, x, features, eps)
    end = timer()
    time_second_zono = end - start
    print(f"TIME SECOND ZONO: {time_second_zono}")
    print(zono_hess)

    with open(f'interactions{num_interactions}/time_second.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_second_zono, time_second_interval])

    plot_second_order(hes_data, hdi_hess, zono_hess)

if __name__ == '__main__':
    main()

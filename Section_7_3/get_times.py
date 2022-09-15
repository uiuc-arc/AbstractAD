import torch

for domain in ['zonos', 'intervals']:
    print(f'{domain} runtimes:')

    l3 = torch.load(f'results/time_{domain}_3layer.pth')
    l4 = torch.load(f'results/time_{domain}_4layer.pth')
    l5 = torch.load(f'results/time_{domain}_5layer.pth')
    lbig = torch.load(f'results/time_{domain}_big.pth')
    print('3layer:', sum(l3)/len(l3))
    print('4layer:', sum(l4)/len(l4))
    print('5layer:', sum(l5)/len(l5))
    print('big:', sum(lbig)/len(lbig))
    print()

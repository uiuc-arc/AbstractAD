import pandas as pd
from scipy.stats import gmean

folder = 'interactions5'
first = pd.read_csv(f'{folder}/first_order.csv', header=None)
second = pd.read_csv(f'{folder}/second_order.csv', header=None)
first_time = pd.read_csv(f'{folder}/time_first.csv', header=None)
second_time = pd.read_csv(f'{folder}/time_second.csv', header=None)

print('Jacobian entries')
a = first.apply(gmean, axis=0)
for n in a:
    print(round(n,2), end=', ')

print('Hessian entries')
b = second.apply(gmean, axis=0)
for n in b:
    print(round(n,2), end=', ')

print(first_time.mean())
print(second_time.mean())

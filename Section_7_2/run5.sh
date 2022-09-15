#!/bin/sh

for i in 1 2 3 4 5
do
   python3 experiments.py --seed $i --network-file interactions5/networks/net$i.pt
done

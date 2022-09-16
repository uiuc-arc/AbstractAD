Abstract Higher-Order Automatic Differentiation
==========
This repository contains the implementation for the paper "A General Construction for Abstract Interpretation of Higher-Order Automatic Differentiation" (OOPSLA 2022) by Jacob Laurel, Rem Yang, Shubham Ugare, Robert Nagel, Gagandeep Singh, and Sasa Misailovic. For both the interval and zonotope domains, we implement abstract first- and second-order automatic differentiation. We use our technique to study (1) robustly explaining a neural network via their first and second derivatives and (2) computing the Lipschitz constant of neural networks.

Requirements
-------------------------
The tool itself requires PyTorch and NumPy. To also plot results, Jupyter Notebook, Matplotlib, and Seaborn are also needed. We ran our experiments with the following software versions: python (3.8.8), torch (1.11.0 cpu), and numpy (1.22.4).

Directory Structure
-------------------------
- **src/**: Contains the core source code.
- **Section_7_2/**: Contains the code for reproducing our results in Section 7.2 of our paper.
- **Section_7_3/**: Contains the code for reproducing our results in Section 7.3 of our paper.

More details for each directory are below.

Source Code
-------------------------
``HyperDuals.py`` contains the interval arithmetic instantiation of our paper’s generic construction for both first and second derivatives. ``Duals.py`` is merely the ``HyperDuals.py`` file with second derivative details removed (to ensure faster runtime if only first-order information is needed). This interval arithmetic instantiation is used as a baseline (as prior work has performed forward-mode automatic differentiation with intervals).  ``HyperDualZono.py`` and  ``SimpleZono.py`` contain the zonotope instantiation of our construction for both 1st/2nd derivatives and just 1st derivatives, respectively.

Section 7.2
-------------------------
In the ``Section_7_2/`` subfolder, running the following command will replicate all results:
```
./run5.sh
```
This runs our robust explanation experiments, ``experiments.py``, on five different seeds and on the networks we have pretrained. Values in Tables 1 and 2 are from the geometric mean of these five runs. Figs. 6 and 7 are the analyses from the experiment with `seed = 2`.

**Individual experiment**  
To run an individual experiment, you can execute:
```
python experiments.py --seed <seed> [--network-file filepath]
```
- ``seed``: an integer seeding the RNGs (required).
- ``filepath``: path to a saved neural network (optional). If unspecified, trains and saves a neural network before analyzing it.

**Analyzing results**  
To obtain the quantitative results (as in Tables 1 and 2), you may run the ``analyze_interpretations.py`` script. The qualitative results (as in Figs. 6 and 7) are automatically outputted as ``.png`` files.

Section 7.3
-------------------------
In the ``Section_7_3/`` subfolder, running the following command will replicate all results:
```
./lipschitz.sh
```
This runs our Lipschitz analysis, ``get_lipschitz.py``, on the 3/4/5-layer and FFNNBig networks we have pretrained. The results (i.e., the computed Lipschitz constants and runtimes) will be saved in the ``results/`` directory with a ``.pth`` extension.

**Individual experiment**  
To only run an experiment on a single network, you may execute:
```
python get_lipschitz.py --network <network name>
```
where ``<network name>`` is either `3layer`, `4layer`, `5layer`, or `big`.

**Plotting**  
To plot the results, run the ``Plot.ipynb`` notebook.
  
**Training and Testing Neural Networks**  
 To train a network from scratch, run: ``python train_mnist.py --network <network name>``. Trained networks will be saved in the ``trained/`` directory. Afterwards, to check the accuracy of the networks on the test set, run: ``python test_mnist.py --network <network name>``. These should all be around or above 98% accuracy. The indices of the correctly classified images will be saved in the ``trained/`` directory as well. Retraining the networks from scratch then verifying their Lipschitz constants should yield very similar – but not exactly the same – results as in the paper (as the training process is stochastic in nature).

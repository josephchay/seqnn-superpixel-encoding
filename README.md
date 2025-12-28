# SEQNN

## Research Papers

Hybrid Quantum Deep Learning with Superpixel Encoding for Earth Observation Data Classification

This paper introduces a hybrid quantum deep learning model (SEQNN) that effectively encodes and analyzes EO data for classification tasks. The proposed model utilizes an efficient encoding approach called superpixel encoding, which reduces the quantum resources required for large image representation by incorporating the concept of superpixels. To validate its effectiveness, we conducted evaluations on multiple EO benchmarks, including Overhead-MNIST, So2Sat LCZ42, and SAT-6 datasets. The experimental results suggest the validity of our model for accurate classification of EO data.

## Datasets

- Overhead-MNIST: https://syncandshare.lrz.de/getlink/fiXGu9L5yBb8Sk8yuHnEnq/ (outdated)
- So2Sat LCZ42: https://syncandshare.lrz.de/getlink/fiKfantyTRUVgeowgQzcAK/ (outdated)
- SAT-6: https://syncandshare.lrz.de/getlink/fi69y5zbjG9SNAeknfPPwQ/ (outdated)
- CIFAR-10/CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html

## Installation

For native Windows: Use SimulatedQuantumLayer (classical mode) - it's GPU-accelerated via PyTorch and works perfectly on Windows.
For quantum GPU on Windows: Use WSL2 (Ubuntu) where you can install the Linux packages:

### Requirements

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Train with default settings (synthetic data, classical mode)
python {v{version}}/SEQNN.py

# Train on CIFAR-10 dataset
python v3/SEQNN.py --dataset cifar10 --epochs 200

# Quick test (10 epochs)
python v3/SEQNN.py --dataset synthetic --quick-test

# Use quantum simulation (slower)
python v3/SEQNN.py --dataset sat --quantum

# Specify custom seed and learning rate
python v3/SEQNN.py --seed 123 --lr 0.001 --batch-size 32

# Evaluate a trained model
python v3/SEQNN.py --eval-only --load-model models/sat/v1/model_final.pt
```

## Conversions

### Jupyter to Python

```bash
jupytext --to py --update {file}.ipynb
```

### Python to Jupyter

```bash
jupytext --to ipynb --update {file}.py
```

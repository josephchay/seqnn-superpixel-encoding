# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Addded
- `.gitignore` file.
- Version 2 of the SEQNN notebook, but results from the training has low accuracy for CIFAR-10 dataset.
- Index-url for the `requirements.txt` file for CUDA 12.6.
- Jax version for the `requirements.txt` file. (Not compatible with CUDA 12.6)
- More details for generated `config.json` file in `models/{dataset}/` directory.

### Changed
- Updated the `load_batch` function in `get_cifar10_data` function in `seqnn_dataLoader.py` file.
- Updated the `requirements.txt` file from original codebase of using TensorFlow to PyTorch.
- Updated `print_args` function in `v4/SEQNN.py` file.

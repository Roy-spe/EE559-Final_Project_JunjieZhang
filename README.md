# EE559 Image Classification Project

This repository compares three neural-network architectures on CIFAR-10 and CIFAR-100:

- **MLP**: a fully connected baseline implemented with **NumPy only**. It does not use PyTorch, TensorFlow, or deep-learning library layers/optimizers.
- **CNN**: a custom convolutional neural network implemented with PyTorch.
- **ResNet-18**: a transfer-learning baseline using torchvision's pretrained ResNet-18.

The project trains each model, evaluates test loss/accuracy/macro-F1, and saves learning curves plus confusion matrices.

## Project Structure

```text
.
|-- data/                         # CIFAR-10/CIFAR-100 archives
|-- experiments/run_all.sh         # Example batch run script
|-- models/
|   |-- mlp.py                     # NumPy-only MLP implementation
|   |-- cnn.py                     # PyTorch CNN implementation
|   `-- resnet.py                  # PyTorch/torchvision ResNet-18
|-- outputs/                       # Saved metrics, plots, checkpoints
|-- utils/
|   |-- numpy_dataset.py           # NumPy CIFAR loader for MLP
|   |-- dataset.py                 # torchvision dataloaders for CNN/ResNet
|   |-- metrics.py                 # accuracy, macro-F1, confusion matrix
|   `-- train_utils.py             # plotting and PyTorch train/eval helpers
|-- train.py                       # Main training entry point
`-- requirements.txt
```

## Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For the **NumPy MLP only**, the required packages are:

```bash
pip install numpy matplotlib scikit-learn
```

PyTorch and torchvision are only required for the CNN and ResNet-18 experiments.

## Data

The project expects CIFAR archives under `data/`:

```text
data/cifar-10-python.tar.gz
data/cifar-100-python.tar.gz
```

The NumPy MLP path reads directly from these archives using `utils/numpy_dataset.py`, so it does not require `torchvision.datasets`.

## Running Experiments

### NumPy MLP

Run CIFAR-10 MLP:

```bash
python train.py --dataset cifar10 --model mlp --epochs 20 --batch_size 128
```

Run CIFAR-100 MLP:

```bash
python train.py --dataset cifar100 --model mlp --epochs 20 --batch_size 128
```

For a faster test run:

```bash
python train.py --dataset cifar10 --model mlp --epochs 1 --batch_size 512 --no_augmentation
```

### PyTorch CNN

```bash
python train.py --dataset cifar10 --model cnn
python train.py --dataset cifar100 --model cnn
```

### PyTorch ResNet-18

```bash
python train.py --dataset cifar10 --model resnet18
python train.py --dataset cifar100 --model resnet18
```

## Command-Line Arguments

`train.py` supports:

```text
--dataset          cifar10 or cifar100
--model            mlp, cnn, or resnet18
--batch_size       training batch size
--epochs           maximum number of epochs
--lr               learning rate
--weight_decay     L2 regularization strength
--val_ratio        validation split ratio
--num_workers      PyTorch dataloader workers, used by CNN/ResNet
--seed             random seed
--patience         early-stopping patience
--no_augmentation  disable random crop/flip augmentation
```

## Outputs

Each run writes files to:

```text
outputs/<dataset>_<model>/
```

Typical output files include:

- `results.txt`: final test loss, accuracy, and macro-F1
- `loss_curve.png`: training and validation loss curve
- `accuracy_curve.png`: training and validation accuracy curve
- `confusion_matrix.png`: test-set confusion matrix
- `best_model.npz`: NumPy MLP checkpoint
- `best_model.pt`: PyTorch CNN/ResNet checkpoint

Merged comparison figures are saved in `outputs/` and can be regenerated with:

```bash
python merge_figures.py
```

## Current Results

| Dataset | Model | Test Loss | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|---:|
| CIFAR-10 | MLP (NumPy) | 1.6752 | 0.4199 | 0.4120 |
| CIFAR-10 | CNN | 0.5879 | 0.7966 | 0.7962 |
| CIFAR-10 | ResNet-18 | 0.5403 | 0.8197 | 0.8187 |
| CIFAR-100 | MLP | 3.8328 | 0.1181 | 0.1005 |
| CIFAR-100 | CNN | 2.1337 | 0.4353 | 0.4223 |
| CIFAR-100 | ResNet-18 | 1.6925 | 0.5529 | 0.5443 |

Note: the CIFAR-10 MLP result above is from the latest NumPy-only run. Older output folders may still contain previous PyTorch checkpoints such as `best_model.pt`.

## Implementation Notes

The MLP in `models/mlp.py` is implemented from basic numerical operations:

- flattened CIFAR image input
- dense layers with He initialization
- ReLU activation
- inverted dropout
- softmax cross-entropy loss
- manual backpropagation
- Adam optimizer implemented in NumPy
- `.npz` checkpoint save/load

The CNN and ResNet-18 implementations remain PyTorch-based because they are convolutional baselines, while the MLP satisfies the basic-library requirement.


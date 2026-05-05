#!/bin/bash

echo "Running CIFAR-10 (MLP)"
python train.py --dataset cifar10 --model mlp

echo "Running CIFAR-10 (CNN)"
python train.py --dataset cifar10 --model cnn

echo "Running CIFAR-100 (CNN)"
python train.py --dataset cifar100 --model cnn

echo "Running CIFAR-100 (ResNet18)"
python train.py --dataset cifar100 --model resnet18
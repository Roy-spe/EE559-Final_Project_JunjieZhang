import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes: int, pretrained: bool = True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Replace final classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
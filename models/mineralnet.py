# models/mineralnet.py
import torch as torch

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MineralNet(nn.Module):
    """
    Transfer-learning model for mineral image classification.
    Uses EfficientNet-B0 pretrained on ImageNet and replaces the
    classifier head with a new linear layer sized to your dataset.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B0 backbone
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = efficientnet_b0(weights=weights)

        # Replace the classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

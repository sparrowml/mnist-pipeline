from typing import Optional

import torch

from . import constants


class MnistFeatures(torch.nn.Module):
    def __init__(self, weights: Optional[str] = None) -> None:
        """Initialize learned layers"""
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64),
        )
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.encoder(x)


class MnistClassifier(torch.nn.Module):
    def __init__(self) -> None:
        """Initialize learned layers"""
        super().__init__()
        self.features = MnistFeatures()
        self.linear = torch.nn.Linear(constants.FEATURE_DIMENSIONS, 128)
        self.batch_norm = torch.nn.BatchNorm1d(128)
        self.classifier = torch.nn.Linear(128, constants.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.features(x)
        x = torch.reshape(x, (len(x), -1))
        x = torch.relu(self.linear(x))
        if len(x) > 1:
            x = self.batch_norm(x)
        x = self.classifier(x)
        return x

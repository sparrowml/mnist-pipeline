import torch


class MnistFeatures(torch.nn.Module):
    def __init__(self) -> None:
        """Initialize learned layers"""
        # TODO: load weights
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.encoder(x)


class MnistClassifier(torch.nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        """Initialize learned layers"""
        super().__init__()
        self.features = MnistFeatures()
        self.linear = torch.nn.Linear(9216, 128)
        self.batch_norm = torch.nn.BatchNorm1d(128)
        self.classifier = torch.nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.features(x)
        x = torch.reshape(x, (len(x), -1))
        x = torch.relu(self.linear(x))
        if len(x) > 1:
            x = self.batch_norm(x)
        x = self.classifier(x)
        return x

import json
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchmetrics

from .config import MnistConfig
from .model import MnistClassifier
from .dataset import load_dataset


class MnistLightning(pl.LightningModule):
    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self.model = MnistClassifier()
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.dev_accuracy = torchmetrics.Accuracy()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        acc = self.train_accuracy(torch.softmax(yhat, -1), y)
        self.log("train_accuracy", acc, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        x, y = batch
        yhat = self.model(x)
        acc = self.dev_accuracy(torch.softmax(yhat, -1), y)
        self.log("dev_accuracy", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self._learning_rate)


def train_model(
    random_seed: int = MnistConfig.random_seed,
    learning_rate: float = MnistConfig.learning_rate,
    max_epochs: int = MnistConfig.max_epochs,
    feature_weights_path: str = MnistConfig.feature_weights_path,
    accuracy_metric_path: str = MnistConfig.accuracy_metric_path,
) -> None:
    """Train the model and save classifier and feature weights."""
    pl.utilities.seed.seed_everything(random_seed)
    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_epochs=max_epochs)
    train_loader = load_dataset()
    dev_loader = load_dataset(train=False)
    pl_model = MnistLightning(learning_rate)
    trainer.fit(pl_model, train_loader, dev_loader)
    torch.save(pl_model.model.features, feature_weights_path)
    with open(accuracy_metric_path, "w") as f:
        f.write(json.dumps(trainer.validate(pl_model, dev_loader, ckpt_path=None)[0]))

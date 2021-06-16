import json
import os
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchmetrics

from .model import MnistClassifier
from .dataset import load_dataset


class MnistLightning(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
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
        self.log('train_accuracy', acc, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> None:
        x, y = batch
        yhat = self.model(x)
        acc = self.dev_accuracy(torch.softmax(yhat, -1), y)
        self.log('dev_accuracy', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=0.01)


def train_model(data_dir: str) -> None:
    """Train the model and save classifier and feature weights."""
    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_epochs=1)
    train_loader = load_dataset(data_dir)
    dev_loader = load_dataset(data_dir, train=False)
    pl_model = MnistLightning()
    trainer.fit(pl_model, train_loader, dev_loader)
    torch.save(pl_model.model.features, os.path.join(data_dir, "features.pt"))
    with open(os.path.join(data_dir, "accuracy.json"), "w") as f:
        f.write(json.dumps(trainer.validate(pl_model, dev_loader, ckpt_path=None)[0]))

from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.serialization import load
import torchmetrics

from .config import MnistConfig
from .model import MnistClassifier
from .dataset import load_dataset


class MnistLightning(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = MnistConfig.batch_size,
        learning_rate: float = MnistConfig.learning_rate,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self.model = MnistClassifier()
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log("loss", loss)
        self.log(
            "accuracy",
            self.accuracy(torch.softmax(yhat, -1), y),
            prog_bar=True,
            on_step=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        x, y = batch
        yhat = self.model(x)
        self.log("accuracy", self.accuracy(torch.softmax(yhat, -1), y))

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self._learning_rate)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return load_dataset(train=True, batch_size=self._batch_size)


def train_model(
    learning_rate: float = MnistConfig.learning_rate,
    max_epochs: int = MnistConfig.max_epochs,
    batch_size: int = MnistConfig.batch_size,
    random_seed: int = MnistConfig.random_seed,
    model_checkpoint_path: str = MnistConfig.model_checkpoint_path,
) -> None:
    """Train the model and save classifier and feature weights."""
    pl.utilities.seed.seed_everything(random_seed)
    trainer = pl.Trainer(checkpoint_callback=False, max_epochs=max_epochs)
    pl_model = MnistLightning(learning_rate=learning_rate, batch_size=batch_size)
    trainer.fit(pl_model)
    trainer.save_checkpoint(model_checkpoint_path)

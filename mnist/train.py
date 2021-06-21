from typing import Tuple

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb

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
        self.log("train_loss", loss)
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
    learning_rate: float = MnistConfig.learning_rate,
    max_epochs: int = MnistConfig.max_epochs,
    batch_size: int = MnistConfig.batch_size,
) -> None:
    """Train the model and save classifier and feature weights."""
    config = MnistConfig(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    wandb.init(config=config.asdict())
    pl.utilities.seed.seed_everything(config.random_seed)
    logger = WandbLogger(project="mnist-pipeline")
    trainer = pl.Trainer(
        logger=logger, checkpoint_callback=False, max_epochs=config.max_epochs
    )
    train_loader = load_dataset(batch_size=config.batch_size)
    dev_loader = load_dataset(train=False, batch_size=config.batch_size)
    pl_model = MnistLightning(config.learning_rate)
    trainer.fit(pl_model, train_loader, dev_loader)
    torch.save(pl_model.model.features, config.feature_weights_path)


def sagemaker_train(*args, **kwargs) -> None:
    train_model()

import json
import pytorch_lightning as pl

from .config import MnistConfig
from .dataset import load_dataset
from .train import MnistLightning


def evaluate_model(
    model_checkpoint_path: str = MnistConfig.model_checkpoint_path,
    batch_size: str = MnistConfig.batch_size,
    metrics_path: str = MnistConfig.metrics_path,
) -> None:
    """Measure validation accuracy on trained model"""
    pl_model = MnistLightning.load_from_checkpoint(model_checkpoint_path)
    validation_loader = load_dataset(train=False, batch_size=batch_size)
    result = pl.Trainer().validate(pl_model, validation_loader)[0]
    with open(metrics_path, "w") as f:
        f.write(json.dumps(result))

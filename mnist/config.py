import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MnistConfig:
    # Dataset
    num_workers: int = 2
    batch_size: int = 64

    # Model
    num_classes: int = 10
    feature_dimensions: int = 9216

    # Train
    random_seed: int = 12345
    learning_rate: float = 0.045
    max_epochs: int = 1

    # Paths
    data_root: Path = Path(os.environ.get("DATA_ROOT", "./data"))

    @classmethod
    @property
    def raw_directory(self) -> str:
        return str(self.data_root / "raw")

    @classmethod
    @property
    def processed_directory(self) -> str:
        return str(self.data_root / "processed")

    @classmethod
    @property
    def model_checkpoint_path(self) -> str:
        return str(self.data_root / "model.ckpt")

    @classmethod
    @property
    def feature_weights_path(self) -> str:
        return str(self.data_root / "features.pt")

    @classmethod
    @property
    def metrics_path(self) -> str:
        return str(self.data_root / "metrics.json")

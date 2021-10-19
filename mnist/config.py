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
    learning_rate: float = 0.05
    max_epochs: int = 2

    # Paths
    repo_root: str = os.environ.get("REPO_ROOT", ".")
    data_root: Path = Path(os.environ.get("DATA_ROOT", "./data"))
    remote_repo: str = "https://github.com/iterative/dataset-registry"
    remote_path: str = "mnist/raw"

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
    def feature_weights_path(self) -> str:
        return str(self.data_root / "features.pt")

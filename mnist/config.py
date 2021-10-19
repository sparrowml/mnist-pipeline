import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from ruamel.yaml import YAML

load_dotenv()
YAML_OVERRIDES_PATH = os.environ.get("YAML_OVERRIDES", "./params.yaml")
yaml_overrides = YAML().load(open(YAML_OVERRIDES_PATH))


@dataclass
class MnistConfig:
    # Dataset
    num_workers: int = yaml_overrides.get("num_workers", 4)
    batch_size: int = yaml_overrides.get("batch_size", 64)

    # Model
    num_classes: int = 10
    feature_dimensions: int = 9216

    # Train
    random_seed: int = yaml_overrides.get("random_seed", 12345)
    learning_rate: float = yaml_overrides.get("learning_rate", 0.05)
    max_epochs: int = yaml_overrides.get("max_epochs", 2)

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
    def feature_weights_path(self) -> str:
        return str(self.data_root / "features.pt")

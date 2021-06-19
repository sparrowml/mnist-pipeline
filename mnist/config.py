import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", "./data")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.yaml")
YAML_CONFIG = dict()
if CONFIG_PATH and os.path.exists(CONFIG_PATH):
    YAML_CONFIG = OmegaConf.load(CONFIG_PATH)


def instancer(cls):
    """Access class like an object instance"""
    return cls()


@instancer
@dataclass
class MnistConfig:
    # Dataset
    num_workers: int = YAML_CONFIG.get("num_workers", 4)
    batch_size: int = YAML_CONFIG.get("batch_size", 128)

    # Model
    num_classes: int = YAML_CONFIG.get("num_classes", 10)
    feature_dimensions: int = YAML_CONFIG.get("feature_dimensions", 9216)

    # Train
    random_seed: int = YAML_CONFIG.get("random_seed", 12345)
    learning_rate: float = YAML_CONFIG.get("learning_rate", 0.01)
    max_epochs: int = YAML_CONFIG.get("max_epochs", 1)

    # Paths
    data_directory: str = DATA_DIRECTORY
    feature_weights_path: str = YAML_CONFIG.get(
        "feature_weights_path", os.path.join(DATA_DIRECTORY, "features.pt")
    )

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

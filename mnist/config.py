import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


@dataclass
class MnistConfig:
    # Dataset
    num_workers: int = 4
    batch_size: int = 64

    # Model
    num_classes: int = 10
    feature_dimensions: int = 9216

    # Train
    random_seed: int = 12345
    learning_rate: float = 0.05
    max_epochs: int = 3

    # Paths
    project_directory: str = str(DATA_ROOT.parent.absolute())
    data_root: str = str(DATA_ROOT.absolute())
    raw_directory: str = str(DATA_ROOT / "raw")
    processed_directory: str = str(DATA_ROOT / "processed")
    feature_weights_path: str = str(DATA_ROOT / "features.pt")

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = Path(os.environ.get("DATA_DIRECTORY", "./data"))


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
    project_directory: str = str(DATA_DIRECTORY.parent.absolute())
    data_directory: str = str(DATA_DIRECTORY.absolute())
    feature_weights_path: str = str(DATA_DIRECTORY / "features.pt")
    sagemaker_weights_path: str = "/opt/ml/model/features.pt"

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", "./data")


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
    data_directory: str = DATA_DIRECTORY
    feature_weights_path: str = os.path.join(DATA_DIRECTORY, "features.pt")

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


MnistSweepConfig = dict(
    name="mnist-sweep",
    method="bayes",
    parameters=dict(
        max_epochs=dict(values=[1, 2, 3]),
        learning_rate=dict(
            min=0.0001,
            max=0.1,
        ),
        batch_size=dict(
            values=[64, 128, 256],
        ),
    ),
    metric=dict(
        name="dev_accuracy",
        goal="maximize",
    ),
)

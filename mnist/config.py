import os
from typing import Tuple
from pathlib import Path

import yaml
from dataclasses import dataclass


@dataclass
class MnistConfig:
    # Model
    n_classes: int = 10
    dropout1_rate: float = 0.25
    dropout2_rate: float = 0.5

    # Train
    seed: int = 12345
    batch_size: int = 128
    n_epochs: int = 1
    verbose: bool = True

    # Nuisance parameters
    n_dataset_threads: int = 4
    shuffle_buffer_size: int = 1024

    # Fixed
    image_height: int = 28
    image_width: int = 28
    n_channels: int = 1
    n_train_samples: int = 60000
    n_test_samples: int = 10000
    artifact_directory: str = str(Path.home()/'.mlpipes/mnist')

    @classmethod
    def from_yaml(cls, path: str):
        """Load overrides from a YAML config file."""
        with open(path) as configfile:
            configdict = yaml.load(configfile)
        return cls(**configdict)

    # Derived values
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (
            self.image_height,
            self.image_width,
            self.n_channels,
        )

    @property
    def steps_per_epoch(self) -> int:
        return self.n_train_samples // self.batch_size

    @property
    def validation_steps(self) -> int:
        return self.n_test_samples // self.batch_size

    @property
    def artifact_directory_path(self) -> Path:
        path_string = os.environ.get(
            'ARTIFACT_DIRECTORY', self.artifact_directory)
        return Path(path_string)

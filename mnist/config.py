"""
Config with sensible defaults
"""
import os
from typing import Tuple
from pathlib import Path

import yaml
from dataclasses import dataclass

from .__version__ import __version__

HOME = Path.home()


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

    # Settings
    artifact_directory: str = str(HOME/'.pipes/mnist')
    train_dataset_filename: str = 'train.tfrecord'
    test_dataset_filename: str = 'test.tfrecord'
    model_weights_filename: str = f'mnist-classifier-{__version__}.h5'
    feature_weights_filename: str = f'mnist-features-{__version__}.h5'

    @classmethod
    def from_yaml(cls: 'MnistConfig', path: str) -> 'MnistConfig':
        with open(path) as configfile:
            configdict = yaml.load(configfile)
        kwargs = {}
        for key, value in configdict.items():
            kwargs[key] = os.environ.get(key.upper()) or value
        return MnistConfig(**kwargs)

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
    def artifact_directory_path(self):
        path = Path(self.artifact_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def train_dataset_path(self) -> Path:
        return self.artifact_directory_path/self.train_dataset_filename

    @property
    def test_dataset_path(self) -> Path:
        return self.artifact_directory_path/self.test_dataset_filename

    @property
    def model_weights_path(self) -> Path:
        return self.artifact_directory_path/self.model_weights_filename

    @property
    def feature_weights_path(self) -> Path:
        return self.artifact_directory_path/self.feature_weights_filename

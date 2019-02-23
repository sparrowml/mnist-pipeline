from typing import Union
from pathlib import Path

import tensorflow as tf

from .config import MnistConfig


class Files:
    def __init__(self, config: Union[str, MnistConfig]=MnistConfig()):
        if isinstance(config, str):
            config = MnistConfig.from_yaml(config)
        self._config = config
        self._directory = Path(config.artifact_directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def train_dataset(self) -> str:
        return str(self._directory/self._config.train_dataset_filename)

    @property
    def test_dataset(self) -> str:
        return str(self._directory/self._config.test_dataset_filename)

    @property
    def model_weights(self) -> str:
        return str(self._directory/self._config.model_weights_filename)

    @property
    def feature_weights(self) -> str:
        return str(self._directory/self._config.feature_weights_filename)

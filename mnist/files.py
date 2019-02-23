import os
from typing import Union
from pathlib import Path
from urllib import request

import tensorflow as tf

from .config import MnistConfig


class Files:
    def __init__(self, config: Union[str, MnistConfig]=MnistConfig()):
        if isinstance(config, str):
            config = MnistConfig.from_yaml(config)
        self._config = config
        self._directory = Path(config.artifact_directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _download_file(self, filename: str) -> None:
        path = self._directory/filename
        if not path.exists():
            url = os.path.join(self._config.artifact_url_prefix, filename)
            request.urlretrieve(url, path)

    @property
    def train_dataset(self) -> str:
        return str(self._directory/self._config.train_dataset_filename)

    def download_train_dataset(self) -> str:
        self._download_file(self._config.train_dataset_filename)
        return self.train_dataset

    @property
    def test_dataset(self) -> str:
        return str(self._directory/self._config.test_dataset_filename)

    def download_test_dataset(self) -> str:
        self._download_file(self._config.test_dataset_filename)
        return self.test_dataset

    @property
    def model_weights(self) -> str:
        return str(self._directory/self._config.model_weights_filename)

    def download_model_weights(self) -> str:
        self._download_file(self._config.model_weights_filename)
        return self.model_weights

    @property
    def feature_weights(self) -> str:
        return str(self._directory/self._config.feature_weights_filename)

    def download_feature_weights(self) -> str:
        self._download_file(self._config.feature_weights_filename)
        return self.feature_weights

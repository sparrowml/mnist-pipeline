import os
from typing import Union
from pathlib import Path
from urllib import request

import tensorflow as tf

from .config import MnistConfig
from .__version__ import __version__


class MnistFiles:
    _artifact_url_prefix: str = 'https://s3.amazonaws.com/mlpipes/mnist/'
    _train_dataset_filename: str = 'train.tfrecord'
    _test_dataset_filename: str = 'test.tfrecord'
    _model_weights_filename: str = f'mnist-classifier-{__version__}.h5'
    _feature_weights_filename: str = f'mnist-features-{__version__}.h5'

    def __init__(self, config: Union[str, MnistConfig]=MnistConfig()):
        if isinstance(config, str):
            config = MnistConfig.from_yaml(config)
        self._directory = config.artifact_directory_path
        self._directory.mkdir(parents=True, exist_ok=True)

    def _download_file(self, filename: str) -> None:
        path = self._directory/filename
        if not path.exists():
            url = os.path.join(self._artifact_url_prefix, filename)
            request.urlretrieve(url, path)

    @property
    def train_dataset(self) -> str:
        return str(self._directory/self._train_dataset_filename)

    @property
    def test_dataset(self) -> str:
        return str(self._directory/self._test_dataset_filename)

    @property
    def model_weights(self) -> str:
        return str(self._directory/self._model_weights_filename)

    def download_model_weights(self) -> str:
        self._download_file(self._model_weights_filename)
        return self.model_weights

    @property
    def feature_weights(self) -> str:
        return str(self._directory/self._feature_weights_filename)

    def download_feature_weights(self) -> str:
        self._download_file(self._feature_weights_filename)
        return self.feature_weights

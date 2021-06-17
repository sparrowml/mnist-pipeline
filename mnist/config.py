import os
from typing import Any, Dict

from omegaconf import OmegaConf

from . import constants


def _get_params() -> Dict[str, Any]:
    if os.path.exists(constants.PARAMS_PATH):
        return OmegaConf.load(constants.PARAMS_PATH)
    return dict()


def instancer(cls):
    """Make __getattribute__ work at the class level"""
    return cls()


@instancer
class MnistConfig:
    _env_prefix = "MNIST_"
    _params = _get_params()
    _constants = constants

    # Dataset
    num_workers: int
    batch_size: int

    # Model
    num_classes: int
    feature_dimensions: int

    # Train
    random_seed: int
    learning_rate: float
    max_epochs: int

    # Paths
    data_directory: str
    feature_weights_path: str
    accuracy_metric_path: str

    def __getattribute__(self, name: str) -> Any:
        """Return config value from 1) environmen variables 2) YAML config 3) constants"""
        if name.startswith("_"):
            return super().__getattribute__(name)
        constant_name = name.upper()
        env_variable = f"{self._env_prefix}{constant_name}"
        if env_variable in os.environ:
            return os.environ[env_variable]
        if name in self._params:
            return self._params[name]
        if hasattr(constants, constant_name):
            return getattr(constants, constant_name)
        raise KeyError(f"{name} not found in config.")

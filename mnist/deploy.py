import os
from .config import MnistConfig

from .model import MnistClassifier


def save_feature_extractor(
        model_path: str, save_folder: str, config_path: str
) -> None:
    """Load model weights and save just the feature weights."""
    config = MnistConfig.from_yaml(config_path)
    model = MnistClassifier()
    model.load_weights(model_path)
    model.features.save_weights(os.path.join(
        save_folder, f'mnist-feature-extractor-{config.version}.h5'
    ))

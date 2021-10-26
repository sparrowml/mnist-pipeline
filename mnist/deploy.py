import torch

from .config import MnistConfig
from .train import MnistLightning


def save_features(
    model_checkpoint_path: str = MnistConfig.model_checkpoint_path,
    feature_weights_path: str = MnistConfig.feature_weights_path,
) -> None:
    """Save feature weights in PyTorch format for transfer learning."""
    pl_model = MnistLightning.load_from_checkpoint(model_checkpoint_path)
    torch.save(pl_model.model.features, feature_weights_path)

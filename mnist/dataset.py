from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .config import MnistConfig


def save_datasets() -> None:
    """Write train and test datasets"""
    MNIST(MnistConfig.data_directory, download=True)


def load_dataset(
    train: bool = True,
    batch_size: int = MnistConfig.batch_size,
) -> DataLoader:
    """Create a data loader"""
    config = MnistConfig(batch_size=batch_size)
    dataset = MNIST(
        config.data_directory,
        download=True,
        train=train,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

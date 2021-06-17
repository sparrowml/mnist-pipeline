from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .config import MnistConfig


def save_datasets(data_directory: str = MnistConfig.data_directory) -> None:
    """Write train and test datasets"""
    MNIST(data_directory, download=True)


def load_dataset(
    train: bool = True,
    data_directory: str = MnistConfig.data_directory,
    batch_size: int = MnistConfig.batch_size,
    num_workers: int = MnistConfig.num_workers,
) -> DataLoader:
    """Create a data loader"""
    dataset = MNIST(
        data_directory, download=True, train=train, transform=transforms.ToTensor()
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

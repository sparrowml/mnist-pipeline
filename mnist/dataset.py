from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from . import constants


def save_datasets() -> None:
    """Write train and test datasets"""
    MNIST(constants.DATA_DIRECTORY, download=True)


def load_dataset(train: bool = True) -> DataLoader:
    """Create a data loader"""
    dataset = MNIST(
        constants.DATA_DIRECTORY,
        download=True,
        train=train,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        dataset,
        batch_size=constants.BATCH_SIZE,
        num_workers=constants.NUM_WORKERS,
    )

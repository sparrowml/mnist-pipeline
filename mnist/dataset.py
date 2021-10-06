import glob
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.utils import extract_archive

from .config import MnistConfig


def gunzip_datasets() -> None:
    """Write train and test datasets"""
    gzip_pattern = str(Path(MnistConfig.raw_directory) / "*.gz")
    Path(MnistConfig.processed_directory).mkdir(parents=True, exist_ok=True)
    for file_path in glob.glob(gzip_pattern):
        print(f"Extracting {file_path}")
        extract_archive(file_path, MnistConfig.processed_directory)


def load_dataset(
    train: bool = True,
    batch_size: int = MnistConfig.batch_size,
) -> DataLoader:
    """Create a data loader"""
    config = MnistConfig(batch_size=batch_size)
    dataset = MNIST(
        config.data_root,
        train=train,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

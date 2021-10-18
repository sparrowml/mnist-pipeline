import glob
import os
from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from .config import MnistConfig


class MnistDataset(Dataset):
    def __init__(self, data_path: str, train: bool) -> None:
        holdout_prefix = "train" if train else "t10k"
        image_path = os.path.join(data_path, f"{holdout_prefix}-images-idx3-ubyte")
        label_path = os.path.join(data_path, f"{holdout_prefix}-labels-idx1-ubyte")
        self.images = datasets.mnist.read_image_file(image_path)
        self.labels = datasets.mnist.read_label_file(label_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple:
        x = self.images[index][None] / 255
        y = self.labels[index]
        return x, y


def gunzip_datasets() -> None:
    """Write train and test datasets"""
    gzip_pattern = str(Path(MnistConfig.raw_directory) / "*.gz")
    Path(MnistConfig.processed_directory).mkdir(parents=True, exist_ok=True)
    for file_path in glob.glob(gzip_pattern):
        print(f"Extracting {file_path}")
        datasets.utils.extract_archive(file_path, MnistConfig.processed_directory)


def load_dataset(
    train: bool = True,
    batch_size: int = MnistConfig.batch_size,
) -> DataLoader:
    """Create a data loader"""
    config = MnistConfig(batch_size=batch_size)
    dataset = MnistDataset(config.processed_directory, train=train)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

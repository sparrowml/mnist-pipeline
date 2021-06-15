from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def save_datasets(data_dir: str) -> None:
    """Write train and test datasets"""
    MNIST(data_dir, download=True)


def load_dataset(data_dir: str, train: bool = True) -> DataLoader:
    """Create a data loader"""
    dataset = MNIST(
        data_dir,
        download=True,
        train=train,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=128, num_workers=2)

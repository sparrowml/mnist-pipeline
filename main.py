import fire

from mnist.dataset import gunzip_datasets
from mnist.train import train_model


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "gunzip-datasets": gunzip_datasets,
            "train-model": train_model,
        }
    )


if __name__ == "__main__":
    main()

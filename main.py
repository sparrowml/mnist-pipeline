import fire

from mnist.dataset import download_dataset, gunzip_dataset
from mnist.model import deploy_model
from mnist.train import train_model


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "download-dataset": download_dataset,
            "gunzip-dataset": gunzip_dataset,
            "deploy-model": deploy_model,
            "train-model": train_model,
        }
    )


if __name__ == "__main__":
    main()

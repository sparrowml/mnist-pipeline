import fire

from mnist.dataset import pull_dataset, gunzip_dataset
from mnist.train import train_model


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "pull-dataset": pull_dataset,
            "gunzip-dataset": gunzip_dataset,
            "train-model": train_model,
        }
    )


if __name__ == "__main__":
    main()

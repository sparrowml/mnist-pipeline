import fire

from mnist.dataset import gunzip_dataset
from mnist.deploy import save_features
from mnist.evaluate import evaluate_model
from mnist.train import train_model


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "evaluate-model": evaluate_model,
            "gunzip-dataset": gunzip_dataset,
            "save-features": save_features,
            "train-model": train_model,
        }
    )


if __name__ == "__main__":
    main()

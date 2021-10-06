import fire

from .dataset import gunzip_datasets
from .train import train_model, launch_sagemaker_train, run_sagemaker_train


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "gunzip-datasets": gunzip_datasets,
            "train-model": train_model,
            "launch-sagemaker-train": launch_sagemaker_train,
            "run-sagemaker-train": run_sagemaker_train,
        }
    )


if __name__ == "__main__":
    main()

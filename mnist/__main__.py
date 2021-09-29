import fire

from .dataset import gunzip_datasets
from .train import train_model, launch_sagemaker_train, run_sagemaker_train
from .sweep import start_sweep, launch_agent


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "gunzip-datasets": gunzip_datasets,
            "train-model": train_model,
            "launch-sagemaker-train": launch_sagemaker_train,
            "run-sagemaker-train": run_sagemaker_train,
            "start-sweep": start_sweep,
            "launch-agent": launch_agent,
        }
    )


if __name__ == "__main__":
    main()

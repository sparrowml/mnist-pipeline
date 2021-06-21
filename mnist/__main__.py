import fire

from .dataset import save_datasets
from .train import train_model, sagemaker_train
from .sweep import start_sweep, launch_agent


def main():
    """Expose CLI functions."""
    fire.Fire(
        {
            "save-datasets": save_datasets,
            "train-model": train_model,
            "sagemaker-train": sagemaker_train,
            "start-sweep": start_sweep,
            "launch-agent": launch_agent,
        }
    )


if __name__ == "__main__":
    main()

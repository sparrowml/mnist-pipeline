import fire

from .dataset import build_dataset
from .train import train_model
from .deploy import save_feature_extractor


def main():
    """Expose CLI functions."""
    fire.Fire({
        'build-dataset': build_dataset,
        'train-model': train_model,
        'save-feature-extractor': save_feature_extractor,
    })

if __name__ == '__main__':
    main()

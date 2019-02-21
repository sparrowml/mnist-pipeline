import fire

from mnist.dataset import build_dataset
from mnist.train import train_model
from mnist.deploy import save_feature_extractor

if __name__ == '__main__':
    fire.Fire()

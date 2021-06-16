import os

# Dataset
NUM_WORKERS: int = 2
BATCH_SIZE: int = 128

# Train
RANDOM_SEED: int = 12345
MAX_EPOCHS: int = 1

# Model
FEATURE_DIMENSIONS: int = 9216
NUM_CLASSES: int = 10

# Paths
PARAMS_PATH: str = "./params.yaml"
DATA_DIRECTORY: str = "./data"
FEATURE_WEIGHTS_PATH: str = os.path.join(DATA_DIRECTORY, "features.pt")
ACCURACY_METRIC_PATH: str = os.path.join(DATA_DIRECTORY, "accuracy.json")

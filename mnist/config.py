import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


@dataclass
class MnistConfig:
    # Dataset
    num_workers: int = 4
    batch_size: int = 64

    # Model
    num_classes: int = 10
    feature_dimensions: int = 9216

    # Train
    random_seed: int = 12345
    learning_rate: float = 0.05
    max_epochs: int = 3

    # Paths
    project_directory: str = str(DATA_ROOT.parent.absolute())
    data_root: str = str(DATA_ROOT.absolute())
    raw_directory: str = str(DATA_ROOT / "raw")
    processed_directory: str = str(DATA_ROOT / "processed")
    feature_weights_path: str = str(DATA_ROOT / "features.pt")

    # SageMaker
    ecr_image: str = "537534971119.dkr.ecr.us-east-1.amazonaws.com/mnist-pipeline"
    branch_name: str = "main"
    instance_count: int = 1
    instance_type: str = "ml.m4.xlarge"
    max_run_duration: str = 3600
    model_output_path: str = "s3://sparrowcomputing/sagemaker/"
    sagemaker_weights_path: str = str(SM_MODEL_DIR / "features.pt")
    sagemaker_execution_role: str = os.getenv("SM_EXECUTION_ROLE")

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

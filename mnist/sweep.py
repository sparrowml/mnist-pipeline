# import wandb
# import yaml

# from .config import MnistConfig
# from .train import train_model


# def start_sweep(config: str) -> str:
#     with open(config) as f:
#         sweep_config = yaml.safe_load(f.read())
#     sweep_id = wandb.sweep(sweep_config)
#     return sweep_id


# def launch_agent(sweep_id: str, count: int = 3) -> None:
#     def train() -> None:
#         with wandb.init():
#             config = MnistConfig(**wandb.config)
#             train_model(config.learning_rate, config.max_epochs, config.batch_size)

#     wandb.agent(sweep_id, function=train, count=count)

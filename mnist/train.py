from typing import Union

import tensorflow as tf

from .config import MnistConfig
from .files import Files
from .dataset import load_dataset
from .model import MnistClassifier


def train_model(config: Union[str, MnistConfig]=MnistConfig()) -> None:
    """Train the model and save classifier and feature weights."""
    if isinstance(config, str):
        config = MnistConfig.from_yaml(config)
    files = Files(config)
    x_train, y_train = load_dataset(files.train_dataset, config)
    x_test, y_test = load_dataset(files.test_dataset, config)
    model = MnistClassifier(config)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adagrad(),
        metrics=['accuracy'],
        target_tensors=[y_train],
    )
    model.fit(
        x_train, y_train,
        verbose=config.verbose,
        epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
    )
    _, accuracy = model.evaluate(
        x_test, y_test,
        verbose=config.verbose,
        steps=config.validation_steps,
    )
    model.save_weights(files.model_weights, overwrite=True)
    model.features.save_weights(files.feature_weights, overwrite=True)
    return f'Test accuracy: {accuracy}'

"""
Given a compiled dataset and a model, this module is responsible for training the model.
"""
from typing import Optional

import tensorflow as tf

from .config import MnistConfig
from .dataset import load_dataset
from .preprocess import preprocess
from .model import classifier


def train(
        train_path: str, test_path: str,
        save_path: Optional[str] = None,
        config_path: Optional[str] = None,
) -> Optional[tf.keras.Model]:
    """Train the model."""
    if config_path:
        config = MnistConfig.from_yaml(config_path)
    else:
        config = MnistConfig()
    x_train, y_train = preprocess(*load_dataset(train_path))
    x_test, y_test = preprocess(*load_dataset(test_path))
    model = classifier(x_train)
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
    print(f'Test accuracy: {accuracy:0.3f}')
    if save_path:
        model.save_weights(
            save_path,
            overwrite=True,
        )
        return
    return model

"""
Given a compiled dataset and a model, this module is responsible for training the model.
"""
from typing import Optional

import tensorflow as tf

from .dataset import load_dataset
from .model import classifier


def train(train_path: str, test_path: str, save_path: Optional[str] = None) -> Optional[tf.keras.Model]:
    """Train the model."""
    x_train, y_train = load_dataset(train_path)
    x_test, y_test = load_dataset(test_path)
    model = classifier(x_train)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adagrad(),
        metrics=['accuracy'],
        target_tensors=[y_train],
    )
    model.fit(
        x_train, y_train,
        verbose=1,
        validation_data=(x_test, y_test),
        steps_per_epoch=60000 // 128,
    )
    _loss, _accuracy = model.evaluate(
        tf.keras.Input(tensor=x_test), y_test,
        verbose=1,
        steps=1,
    )
    print('Test loss:', _loss)
    print('Test accuracy:', _accuracy)
    if save_path:
        model.save(save_path)
        return
    return model

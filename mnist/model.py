"""
Define the structure of the model.
"""
from typing import Tuple, Union

import tensorflow as tf

from .config import MnistConfig


def feature_extractor(images: tf.keras.Input) -> tf.Tensor:
    """Define feature extractor."""
    hidden = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
    )(images)
    hidden = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
    )(hidden)
    hidden = tf.keras.layers.MaxPool2D()(hidden)
    hidden = tf.keras.layers.Dropout(MnistConfig.dropout1_rate)(hidden)
    return hidden


def classifier(images: tf.keras.Input) -> tf.keras.Model:
    """Define classifier."""
    hidden = feature_extractor(images)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(MnistConfig.dropout2_rate)(hidden)
    predictions = tf.keras.layers.Dense(MnistConfig.n_classes, activation='softmax')(hidden)
    return tf.keras.Model(inputs=images, outputs=predictions)

"""
Define the structure of the model.
"""
from typing import Tuple, Union

import tensorflow as tf


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
    hidden = tf.keras.layers.Dropout(0.25)(hidden)
    return hidden


def classifier(images: tf.keras.Input) -> tf.keras.Model:
    """Define classifier."""
    hidden = feature_extractor(images)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(hidden)
    return tf.keras.Model(inputs=images, outputs=predictions)

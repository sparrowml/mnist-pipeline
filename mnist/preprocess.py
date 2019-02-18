from typing import Tuple

import tensorflow as tf

from .config import MnistConfig


def preprocess(images: tf.Tensor, label: tf.Tensor) -> Tuple[tf.keras.Input, tf.Tensor]:
    """Preprocess image and label."""
    images = tf.to_float(images) / 255
    return tf.keras.Input(tensor=images), tf.one_hot(label, MnistConfig.n_classes)

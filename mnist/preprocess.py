from typing import Tuple, Optional

import tensorflow as tf

from .config import MnistConfig


def preprocess(
        images: tf.Tensor, label: tf.Tensor,
        config: Optional[MnistConfig] = MnistConfig()
) -> Tuple[tf.keras.Input, tf.Tensor]:
    """Preprocess images and labels."""
    images = tf.to_float(images) / 255
    return tf.keras.Input(tensor=images), tf.one_hot(label, config.n_classes)

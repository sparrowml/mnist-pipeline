from typing import Tuple

import tensorflow as tf


def preprocess(images: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Preprocess image and label."""
    return tf.to_float(images) / 255, tf.one_hot(label, 10)

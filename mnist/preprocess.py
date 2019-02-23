import tensorflow as tf


def preprocess_images(images: tf.Tensor) -> tf.Tensor:
    """Map image values to [0, 1]."""
    return images / 255


def preprocess_labels(labels: tf.Tensor, n_classes: int) -> tf.Tensor:
    """Generate one-hot encoding for class labels."""
    return tf.one_hot(labels, n_classes)

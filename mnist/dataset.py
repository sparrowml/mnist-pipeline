"""
Compile a TFRecord dataset for training. Each example has the following form:

{
    "image": <raw image bytes>,
    "label": <class label>
}
"""
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import MnistConfig
from .preprocess import preprocess


def _int64_feature(value: int) -> tf.train.Feature:
    """int64 feature wrapper"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value: bytes) -> tf.train.Feature:
    """bytes feature wrapper"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _parse_example(example: tf.train.Example) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode a TFRecord example"""
    features = dict(
        image=tf.FixedLenFeature([], tf.string),
        label=tf.FixedLenFeature([], tf.int64),
    )
    parsed = tf.parse_single_example(example, features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    label = parsed['label']
    return image, label


def _write_holdout(images: np.ndarray, labels: np.ndarray, file_path: str) -> None:
    """Write a single TFRecord file for either train or test."""
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for x, y in zip(images, labels):
            feature = dict(
                image=_bytes_feature(x.tostring()),
                label=_int64_feature(int(y)),
            )
            example = tf.train.Example(
                features=tf.train.Features(feature=feature),
            )
            writer.write(example.SerializeToString())


def build_dataset(train_path: str, test_path: str) -> None:
    """Write train and test TFRecord files."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    _write_holdout(x_train, y_train, train_path)
    _write_holdout(x_test, y_test, test_path)


def load_dataset(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Create a dataset generator from a TFRecord path."""
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(
        _parse_example,
        num_parallel_calls=MnistConfig.n_dataset_threads,
    )
    dataset = dataset.shuffle(MnistConfig.shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(MnistConfig.batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    images = tf.reshape(images, [-1, *MnistConfig.image_shape()])
    return images, labels

"""
Functionality for dealing with the TFRecord dataset. Each record has the
following format:

{
    "image": <raw image bytes>,
    "label": <class label>
}
"""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

from .config import MnistConfig
from .files import MnistFiles
from .preprocess import preprocess_images, preprocess_labels


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


def _write_holdout(
        images: np.ndarray, labels: np.ndarray, file_path: str
) -> None:
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


def save_datasets(config: Union[str, MnistConfig]=MnistConfig()) -> None:
    """Write train and test TFRecord files."""
    if isinstance(config, str):
        config = MnistConfig.from_yaml(config)
    files = MnistFiles(config)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    _write_holdout(
        x_train[:config.n_train_samples],
        y_train[:config.n_train_samples],
        files.train_dataset,
    )
    _write_holdout(
        x_test[:config.n_test_samples],
        y_test[:config.n_test_samples],
        files.test_dataset,
    )


def load_dataset(
    file_path: Path, config: Union[str, MnistConfig]=MnistConfig()
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Create a dataset generator from a TFRecord path."""
    if isinstance(config, str):
        config = MnistConfig.from_yaml(config)
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(
        _parse_example,
        num_parallel_calls=config.n_dataset_threads,
    )
    dataset = dataset.shuffle(config.shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(config.batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    images = tf.reshape(images, [-1, *config.image_shape])
    return (
        preprocess_images(images),
        preprocess_labels(labels, config.n_classes),
    )

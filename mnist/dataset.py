"""
Functionality for dealing with the TFRecord dataset. Each record has the
following format:

{
    "image": <raw jpeg bytes>,
    "label": <class label>
}
"""
import io
from pathlib import Path
from typing import Tuple, Union, List, Dict

import imageio
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
    image = tf.image.decode_jpeg(parsed['image'])
    label = parsed['label']
    return image, label


def write_holdout(samples: List[Dict], file_path: str) -> None:
    """Write a single TFRecord file for either train or test."""
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for sample in samples:
            buffer = io.BytesIO()
            img = sample['img']
            imageio.imwrite(buffer, img, format='jpg')
            buffer.seek(0)
            feature = dict(
                image=_bytes_feature(buffer.read()),
                label=_int64_feature(int(sample['label'])),
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
    train_arrays, test_arrays = tf.keras.datasets.mnist.load_data()
    train_samples = [
        dict(img=img, label=int(label)) for img, label in zip(*train_arrays)]
    test_samples = [
        dict(img=img, label=int(label)) for img, label in zip(*test_arrays)]
    write_holdout(train_samples[:config.n_train_samples], files.train_dataset)
    write_holdout(test_samples[:config.n_test_samples], files.test_dataset)


def load_dataset(
        file_path: Path, config: MnistConfig
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Create a dataset generator from a TFRecord path."""
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

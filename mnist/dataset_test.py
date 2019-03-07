import unittest

import tensorflow as tf

from .config import MnistConfig
from .files import MnistFiles
from .dataset import load_dataset


class TestDataset(unittest.TestCase):
    def test_load_dataset__generates_correct_shape(self):
        config = MnistConfig()
        files = MnistFiles()
        sess = tf.InteractiveSession()
        images_tensor, labels_tensor = load_dataset(files.train_dataset)
        images, labels = sess.run([images_tensor, labels_tensor])
        self.assertEqual(
            images.shape, (config.batch_size, *config.image_shape)
        )
        self.assertEqual(labels.shape, (config.batch_size, config.n_classes))

import unittest

import numpy as np

from .config import MnistConfig
from .model import MnistClassifier, MnistFeatures


class TestModel(unittest.TestCase):
    def test_classifier_load_pretrained_weights__changes_weights(self):
        model = MnistClassifier()
        x = np.random.uniform(size=(1, 28, 28, 1))
        random_y = model.predict(x)
        model.load_pretrained_weights()
        trained_y = model.predict(x)
        self.assertFalse((random_y == trained_y).all())

    def test_classifier__predicts_n_class_array(self):
        # NOTE: this would make loading pretrained weights fail
        config = MnistConfig(n_classes=99)
        model = MnistClassifier(config)
        x = np.random.uniform(size=(1, 28, 28, 1))
        self.assertEqual(model.predict(x).shape, (1, 99))

    def test_classifier__breaks_on_different_image_sizes(self):
        model = MnistClassifier()
        model.load_pretrained_weights()
        x = np.random.uniform(size=(1, 32, 32, 1))
        self.assertRaises(ValueError, lambda: model.predict(x))

    def test_features_load_pretrained_weights__changes_weights(self):
        model = MnistFeatures()
        x = np.random.uniform(size=(1, 28, 28, 1))
        random_y = model.predict(x)
        model.load_pretrained_weights()
        trained_y = model.predict(x)
        self.assertFalse((random_y == trained_y).all())

    def test_features__works_on_different_image_sizes(self):
        model = MnistFeatures()
        model.load_pretrained_weights()
        x1 = np.random.uniform(size=(1, 32, 32, 1))
        _ = model.predict(x1)
        x2 = np.random.uniform(size=(1, 64, 64, 1))
        _ = model.predict(x2)

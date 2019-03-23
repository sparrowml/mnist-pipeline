import unittest

import numpy as np

from .config import MnistConfig
from .model import mnist_features, mnist_classifier


class TestModel(unittest.TestCase):
    def test_classifier__predicts_n_class_array(self):
        # NOTE: this would make loading pretrained weights fail
        config = MnistConfig(n_classes=99)
        model = mnist_classifier(config)
        x = np.random.uniform(size=(1, 28, 28, 1))
        self.assertEqual(model.predict(x).shape, (1, 99))

    def test_classifier__breaks_on_different_image_sizes(self):
        model = mnist_classifier(pretrained=True)
        x = np.random.uniform(size=(1, 32, 32, 1))
        self.assertRaises(ValueError, lambda: model.predict(x))

    def test_features__works_on_different_image_sizes(self):
        model = mnist_features(pretrained=True)
        x1 = np.random.uniform(size=(1, 32, 32, 1))
        _ = model.predict(x1)
        x2 = np.random.uniform(size=(1, 64, 64, 1))
        _ = model.predict(x2)

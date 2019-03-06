from typing import Union
import tensorflow as tf

from .config import MnistConfig
from .files import MnistFiles


class MnistFeatures(tf.keras.Model):
    """Convolutional stack for MNIST classifier"""
    def __init__(self, config: Union[str, MnistConfig]=MnistConfig()):
        super().__init__()
        if isinstance(config, str):
            config = MnistConfig.from_yaml(config)
        self._config = config
        # Layers
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool = tf.keras.layers.MaxPool2D()
        self.dropout = tf.keras.layers.Dropout(config.dropout1_rate)
        # Build model
        self.call(tf.keras.Input((None, None, config.n_channels)))

    def load_pretrained_weights(self):
        weights_path = MnistFiles(self._config).download_feature_weights()
        self.load_weights(str(weights_path))

    def call(self, images: tf.keras.Input) -> tf.Tensor:
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.pool(x)
        return self.dropout(x)


class MnistClassifier(tf.keras.Model):
    """MNIST digit classifier"""
    def __init__(self, config: Union[str, MnistConfig]=MnistConfig()):
        super().__init__()
        if isinstance(config, str):
            config = MnistConfig.from_yaml(config)
        self._config = config
        # Layers
        self.features = MnistFeatures(config)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(config.dropout2_rate)
        self.dense2 = tf.keras.layers.Dense(
            config.n_classes,
            activation='softmax'
        )
        # Build model
        self.call(tf.keras.Input(config.image_shape))

    def load_pretrained_weights(self):
        weights_path = MnistFiles(self._config).download_model_weights()
        self.load_weights(str(weights_path))

    def call(self, images: tf.keras.Input) -> tf.Tensor:
        x = self.features(images)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

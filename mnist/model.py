from typing import Union
import tensorflow as tf

from .config import MnistConfig
from .files import MnistFiles


def mnist_features(
        config: Union[str, MnistConfig] = MnistConfig(),
        pretrained: bool = False
) -> tf.keras.Model:
    if isinstance(config, str):
        config = MnistConfig.from_yaml(config)
    inputs = tf.keras.Input((None, None, config.n_channels))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    outputs = tf.keras.layers.BatchNormalization()(x)
    features = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='mnist-features',
    )
    if pretrained:
        files = MnistFiles(config)
        features.load_weights(files.download_feature_weights())
    return features


def mnist_classifier(
        config: Union[str, MnistConfig] = MnistConfig(),
        pretrained: bool = False
) -> tf.keras.Model:
    if isinstance(config, str):
        config = MnistConfig.from_yaml(config)
    inputs = tf.keras.Input(config.image_shape)
    x = mnist_features(config)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(
        config.n_classes,
        activation='softmax'
    )(x)
    classifier = tf.keras.Model(inputs=inputs, outputs=outputs)
    if pretrained:
        files = MnistFiles(config)
        classifier.load_weights(files.download_model_weights())
    return classifier

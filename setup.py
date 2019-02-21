from setuptools import setup

from mnist.config import MnistConfig

setup(
    name='mnist',
    version=MnistConfig.version,
    packages=['mnist'],
    license='MIT',
)

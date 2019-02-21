from setuptools import setup

from mnist.config import MnistConfig

setup(
    name='mnist',
    version=MnistConfig.version,
    packages=['mnist'],
    license='MIT',
    install_requires=[
        'dataclasses',
        'fire',
        'numpy',
        'pyyaml',
    ],
    extras_require={
        'cpu': ['tensorflow'],
        'gpu': ['tensorflow-gpu'],
    },
)

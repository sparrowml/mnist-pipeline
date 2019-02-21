import os
from setuptools import setup

directory = os.path.dirname(__file__)
with open(os.path.join(directory, 'mnist', '__version__')) as version_file:
    version = version_file.read().strip()

setup(
    name='mnist',
    version=version,
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

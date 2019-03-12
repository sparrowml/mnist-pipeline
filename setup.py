import os
from setuptools import setup

about = {}
directory = os.path.dirname(__file__)
with open(os.path.join(directory, 'mnist', '__version__.py')) as version:
    exec(version.read(), about)

setup(
    name='mnist-pipeline',
    version=about['__version__'],
    packages=['mnist'],
    license='MIT',
    install_requires=[
        'dataclasses',
        'fire',
        'pyyaml',
    ],
    extras_require={
        'cpu': ['tensorflow'],
        'gpu': ['tensorflow-gpu'],
    },
    entry_points={
        'console_scripts': [
            'mnist = mnist.__main__:main',
        ]
    }
)

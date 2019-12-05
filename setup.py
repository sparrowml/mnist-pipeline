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
        'imageio',
        'pyyaml',
    ],
    extras_require={
        'cpu': ['tensorflow<2.0'],
        'gpu': ['tensorflow-gpu<2.0'],
    },
    entry_points={
        'console_scripts': [
            'mnist = mnist.__main__:main',
        ]
    },
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
    ]
)

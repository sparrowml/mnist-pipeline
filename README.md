# MNIST Classifier Pipeline
A toy AI pipeline that creates a digit classifier with features that can be exported.

```
Raw MNIST Data -> TFRecord Datasets -> Digit Classifier -> Classifier Weights
                                                        -> Feature Extractor Weights
```

## Installation

``` shell
pip install mnist-pipeline[cpu]
```
...or swap in `[gpu]` at the end to use the `tensorflow-gpu` package.

## Inference quick start

Use the `load_pretrained_weights()` method to get a pretrained classifier. Then pass in a batch of TensorFlow or numpy images with shape `(batch_size, 28, 28, 1)` to `predict()`. If the images are 8-bit integers, divide by 255 to map to floats in `[0, 1]`.

``` python
import numpy as np
from mnist import mnist_classifier

x = np.random.randint(
    0, 256,
    size=(1, 28, 28, 1),
    dtype=np.uint8
) / 255

model = mnist_classifier(pretrained=True)

y = model.predict(x).argmax()
```

## Running the pipeline

There are two commands in the pipeline:

1. `save-datasets`
2. `train-model`

Each command can be run with with `mnist` CLI, e.g. `mnist save-datasets`. Optionally, they can also both take the path to a YAML config file with overrides. The values in these files override the defaults set in [mnist/config.py](./mnist/config.py). So for example, to run the pipeline with a batch size of 64 for 2 epochs, you can call `mnist train-model config/pipeline.yml`.

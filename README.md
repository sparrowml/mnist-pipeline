# MNIST Classifier Pipeline
A toy ML pipeline that creates a digit classifier with features that can be exported.

```
  +--------------+
  | data/raw.dvc |
  +--------------+
          *
          *
          *
+-----------------+
| gunzip-datasets |
+-----------------+
          *
          *
          *
  +-------------+
  | train-model |
  +-------------+
```

## Quick Start

You can reproduce the pipeline with the following commands:

```
pip install .
dvc repro
```

## Development

For development, it's strongly recommended to use the remote containers plugin. Build the image and start developing inside the resulting Docker container. You will need to define a `.env` file locally.

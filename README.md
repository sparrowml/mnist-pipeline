# MNIST Classifier Pipeline
MNIST classifier pipeline that creates a digit classifier with features that can be exported.

The pattern of this pipeline:

```
Raw MNIST Data -> TFRecord Datasets -> Digit Classifier -> Classifier Weights
                                                        -> Feature Extractor Weights
```
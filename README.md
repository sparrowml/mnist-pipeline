# MNIST Classifier Pipeline
MNIST classifier pipeline that creates a digit classifier with features that can be exported.

The pattern of this pipeline:
```
MNIST Data -> TFRecord Datasets -> CNN Classifier -> Classifier Weights
                                                  -> Feature Extractor Weights
```
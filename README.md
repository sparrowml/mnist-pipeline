# MNIST Classifier Pipeline
MNIST classifier pipeline that creates a digit classifier with features that can be exported.

The pattern of this pipeline:
```
Public Dataset -> Compiled Training Dataset -> Trained Model  -> Feature Extractor
```

More concretely:
```
MNIST          -> TFRecords                 -> CNN Classifier -> Saved Weights
```

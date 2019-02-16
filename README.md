# MNIST Classifier Pipeline
MNIST transfer learning pipeline that trains an MNIST classifier and creates a model file that can be run in the browser. Although this is a toy example, the batteries are included.

The pattern of this pipeline:
```
Public Dataset -> Compiled Training Dataset -> Trained Model
                                                    |
Public Dataset -> Compiled Training Dataset -> Trained Model -> Inference Weights
```

More concretely:
```
Fashion MNIST -> TFRecords -> CNN Classifier
                                    |
MNIST         -> TFRecords -> CNN Classifier -> TensorFlow.js Model
```

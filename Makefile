build-dataset:
	python main.py build_dataset data/train.tfrecord data/test.tfrecord

train:
	python main.py train data/train.tfrecord data/test.tfrecord data/mnist.h5

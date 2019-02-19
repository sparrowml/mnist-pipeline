build-dataset:
	python main.py build_dataset \
		--train_path data/train.tfrecord \
		--test_path data/test.tfrecord

train:
	python main.py train \
		--train_path data/train.tfrecord \
		--test_path data/test.tfrecord \
		--save_path data/mnist.h5 \
		--config_path config.yaml

build-dataset:
	python main.py build_dataset \
		--train_path data/train.tfrecord \
		--test_path data/test.tfrecord

train-model:
	python main.py train_model \
		--train_path data/train.tfrecord \
		--test_path data/test.tfrecord \
		--save_path data/mnist.h5 \
		--config_path config.yaml

save-feature-extractor:
	python main.py save_feature_extractor \
		--model_path data/mnist.h5 \
		--save_folder data/ \
		--config_path config.yaml

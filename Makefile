build-dataset:
	mnist build-dataset \
		--train_path data/train.tfrecord \
		--test_path data/test.tfrecord

train-model:
	mnist train-model \
		--train_path data/train.tfrecord \
		--test_path data/test.tfrecord \
		--save_path data/mnist.h5 \
		--config_path config.yml

save-feature-extractor:
	mnist save-feature-extractor \
		--model_path data/mnist.h5 \
		--save_folder data/ \
		--config_path config.yml

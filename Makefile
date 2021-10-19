install:
	pip install poetry==1.1.6
	poetry install

import-dataset:
	dvc import https://github.com/iterative/dataset-registry mnist/raw -o data/raw

push-model:
	dvc push -r s3 data/features.pt

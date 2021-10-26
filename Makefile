install:
	pip install poetry==1.1.6
	poetry install

import-dataset:
	dvc import https://github.com/iterative/dataset-registry mnist/raw -o data/raw

pull:
	dvc pull data/raw

push: pipeline
	dvc push -r s3 data/features.pt

pipeline: pull
	dvc repro

install:
	pip install poetry==1.1.6
	poetry install

pull:
	dvc pull

repro:
	dvc repro

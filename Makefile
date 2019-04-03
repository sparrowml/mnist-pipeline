install:
	@pip install -e .[cpu]

test: export ARTIFACT_DIRECTORY=./.test
test:
	# Run the pipeline on a single batch and tests
	@mnist save-datasets ./config/test.yml
	@mnist train-model ./config/test.yml
	@python setup.py test

pipeline:
	@mnist save-datasets ./config/pipeline.yml
	@mnist train-model ./config/pipeline.yml

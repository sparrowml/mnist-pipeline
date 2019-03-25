test: export ARTIFACT_DIRECTORY=./.test
test:
	# Run the pipeline on a single batch and tests
	@tox -e save-datasets ./config/test.yml
	@tox -e train-model ./config/test.yml
	@tox -e pytest

pipeline:
	@tox -e save-datasets ./config/pipeline.yml
	@tox -e train-model ./config/pipeline.yml

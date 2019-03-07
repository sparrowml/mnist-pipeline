test:
	@tox -e save-datasets
	@tox -e train-model ./config/test.yml
	@tox -e pytest

pipeline:
	@tox -e save-datasets ./config/pipeline.yml
	@tox -e train-model ./config/pipeline.yml

deploy:
	@tox -e upload-weights ${ARTIFACT_DIRECTORY}

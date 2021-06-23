FROM python:3.9

RUN pip install poetry==1.1.6

RUN mkdir /code
WORKDIR /code

COPY poetry.lock poetry.toml pyproject.toml /code/

# Create dummy package for poetry installation
RUN mkdir mnist
RUN touch mnist/__init__.py

RUN poetry install
ADD . .

ENTRYPOINT [ "mnist", "run-sagemaker-train" ]

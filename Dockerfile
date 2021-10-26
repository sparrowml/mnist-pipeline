FROM python:3.9

RUN pip install poetry==1.1.6

RUN mkdir /code
WORKDIR /code

RUN mkdir mnist && touch mnist/__init__.py

COPY poetry.lock poetry.toml pyproject.toml /code/
RUN poetry install

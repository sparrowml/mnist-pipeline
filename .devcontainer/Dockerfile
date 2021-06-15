FROM python:3.9

RUN pip install poetry==1.1.6

RUN mkdir /code
WORKDIR /code

COPY poetry.lock poetry.toml pyproject.toml /code/

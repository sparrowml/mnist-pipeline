FROM python:3.9

RUN mkdir /code
WORKDIR /code

ADD mnist mnist
ADD setup.py setup.py

RUN pip install -e .

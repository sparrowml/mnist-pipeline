FROM python:3.6

RUN mkdir /code
WORKDIR /code

ADD mnist mnist
ADD setup.py setup.py

RUN pip install -e .[cpu]

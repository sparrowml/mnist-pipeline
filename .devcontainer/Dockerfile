FROM python:3.9

RUN mkdir /code
WORKDIR /code

RUN pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip install pytorch-lightning torchvision ipykernel

ADD mnist mnist
ADD setup.py setup.py

RUN pip install -e .

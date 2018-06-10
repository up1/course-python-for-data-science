FROM python:3.6.5-onbuild
COPY requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt
COPY . /src
WORKDIR /src

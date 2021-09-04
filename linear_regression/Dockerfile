FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3-pip

WORKDIR /workspace

COPY . /workspace/

RUN pip install sklearn pandas
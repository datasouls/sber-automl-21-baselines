FROM python:3.8.12-buster as base

LABEL PROJECT="Datasouls Competition: AutoGluon"

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
    && rm -rf /var/lib/apt/lists/* &&\
      apt-get clean

FROM base as builder

COPY . .
RUN pip install --upgrade --prefix=/usr/local -r /requirements.txt

WORKDIR /app

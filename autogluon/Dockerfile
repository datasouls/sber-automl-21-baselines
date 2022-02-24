FROM python:3.8.12-buster

LABEL PROJECT="Datasouls Competition: AutoGluon"

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
    && rm -rf /var/lib/apt/lists/* &&\
      apt-get clean

WORKDIR /app

COPY . .
RUN pip install --upgrade --prefix=/usr/local -r ./requirements.txt


FROM python:3.8.12-buster as base

LABEL PROJECT="Datasouls Competition: Autogluon"

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
    && rm -rf /var/lib/apt/lists/* &&\
      apt-get clean

FROM base as builder

WORKDIR /builds
COPY requirements.txt /requirements.txt
RUN pip --no-cache-dir install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip \
 && pip --no-cache-dir install --upgrade wheel setuptools \
 && pip --no-cache-dir install --prefix=/install -r /requirements.txt

FROM base

COPY --from=builder /install /usr/local
WORKDIR /workspace

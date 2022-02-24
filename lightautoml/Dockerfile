FROM debian:buster as base

LABEL PROJECT="Datasouls Competition: LAML"

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
  python3 \
  python3-pip \
    && rm -rf /var/lib/apt/lists/* &&\
      apt-get clean

FROM base as builder

WORKDIR /builds
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install --prefix=/install -r /requirements.txt

FROM base

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY --from=builder /install /usr/local
WORKDIR /workspace

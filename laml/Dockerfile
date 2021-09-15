FROM python:3.8.12-buster as base

LABEL PROJECT="Datasouls Competition: LAML"

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
    && rm -rf /var/lib/apt/lists/* &&\
      apt-get clean

FROM base as builder

WORKDIR /builds

WORKDIR /builds
COPY requirements.txt /requirements.txt
RUN pip install --prefix=/install -r /requirements.txt

FROM base

COPY --from=builder /install /usr/local
WORKDIR /workspace

COPY . /workspace/

ENTRYPOINT ["/bin/bash"]

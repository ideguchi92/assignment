FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  git \
  libopencv-dev \
  python3-pip \
  python3.8 \
  python3.8-dev \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m pip install -U pip

WORKDIR /assignment
RUN git clone --depth=1 https://github.com/NVIDIA/apex.git
RUN git clone --depth=1 https://github.com/ifzhang/ByteTrack.git
RUN git clone --depth=1 -b v1.3.9 https://github.com/open-mmlab/mmcv.git
RUN git clone --depth=1 https://github.com/ViTAE-Transformer/ViTPose.git

RUN python3 -m pip install 'numpy<1.24'
RUN python3 -m pip install -r ByteTrack/requirements.txt
RUN python3 -m pip install -r ViTPose/requirements/build.txt
RUN python3 -m pip install -r apex/requirements_dev.txt

WORKDIR /assignment/apex
RUN python3 -m pip install -v --disable-pip-version-check --no-cache-dir ./

WORKDIR /assignment/ByteTrack
RUN python3 setup.py develop
RUN python3 -m pip install cython
RUN python3 -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN python3 -m pip install cython_bbox

WORKDIR /assignment/mmcv
RUN MMCV_WITH_OPS=1 python3 -m pip install -e .

WORKDIR /assignment/ViTPose
RUN python3 -m pip install -v -e .
RUN python3 -m pip install timm einops

WORKDIR /assignment

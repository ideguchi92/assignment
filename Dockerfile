FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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
RUN git clone --depth=1 https://github.com/gatagat/lap.git
RUN git clone --depth=1 https://github.com/ifzhang/ByteTrack.git
RUN git clone --depth=1 -b v1.3.9 https://github.com/open-mmlab/mmcv.git
RUN git clone --depth=1 https://github.com/ViTAE-Transformer/ViTPose.git
#RUN git clone --depth=1 https://github.com/ashutosh1807/PixelFormer.git
#RUN git clone --depth=1 https://github.com/isl-org/MiDaS.git
#RUN git clone --depth=1 https://github.com/isl-org/DPT.git
#RUN git clone --depth=1 https://github.com/aim-uofa/AdelaiDepth.git

WORKDIR /assignment/lap
RUN python3 -m pip install cython 'numpy<1.24'
RUN python3 setup.py build
RUN python3 setup.py install

WORKDIR /assignment
RUN python3 -m pip install -r ByteTrack/requirements.txt
RUN python3 -m pip install -r ViTPose/requirements/build.txt
#RUN python3 -m pip install -r AdelaiDepth/LeReS/requirements.txt

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

#WORKDIR /assignment/DPT
#RUN python3 -m pip install -v -e .

#ENV PYTHONPATH $PYTHONPATH:/assignment/AdelaiDepth/LeReS/Minist_Test

WORKDIR /assignment

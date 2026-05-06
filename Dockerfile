FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG CMAKE_BUILD_PARALLEL_LEVEL=2

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV LIBTORCH_VERSION=2.4.1+cu118
ENV LIBTORCH_ARCHIVE=libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu118.zip
ENV CMAKE_PREFIX_PATH=/opt/libtorch
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch
ENV Caffe2_DIR=/opt/libtorch/share/cmake/Caffe2
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9
ENV TCNN_CUDA_ARCHITECTURES=75;80;86;89
ENV CMAKE_CUDA_ARCHITECTURES=75;80;86;89
ENV CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        git \
        libdw-dev \
        libeigen3-dev \
        libgl1 \
        libglib2.0-0 \
        libomp-dev \
        libopencv-dev \
        libopenmpi-dev \
        libpcl-dev \
        python3 \
        python3-pip \
        unzip \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install "cmake<4" open3d==0.18.0


RUN mkdir -p /opt \
    && wget -qO- "https://download.pytorch.org/libtorch/cu118/${LIBTORCH_ARCHIVE}" > /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm -f /tmp/libtorch.zip


COPY thirdparty/GS-SDF/docker-build /opt/gs-sdf-build

WORKDIR /opt/gs-sdf-build

RUN cmake -S . -B build \
        -DENABLE_ROS=OFF \
        -DTorch_DIR="${Torch_DIR}" \
        -DTCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
    && cmake --build build


RUN apt-get update \
    && apt-get install -y --no-install-recommends  colmap

WORKDIR /3dgs_pipe/thirdparty/GS-SDF

FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y git python3 python3-pip wget ninja-build mpich libopenmpi-dev openjdk-8-jre-headless
RUN python3 -m pip install "nanobind>=2.0"

RUN mkdir /opt/cmake

WORKDIR /opt/cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.0-rc1/cmake-3.29.0-rc1-linux-x86_64.sh
RUN bash cmake-3.29.0-rc1-linux-x86_64.sh --skip-license
RUN echo "export PATH='/opt/cmake/bin':$PATH" >> ~/.bashrc

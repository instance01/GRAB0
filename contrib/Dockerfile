FROM python:3.8.3-slim-buster

WORKDIR /app

COPY 99defaultrelease /etc/apt/apt.conf.d/
COPY stable.list /etc/apt/sources.list.d/
COPY testing.list /etc/apt/sources.list.d/

RUN printf "deb http://httpredir.debian.org/debian buster-backports main non-free\ndeb-src http://httpredir.debian.org/debian buster-backports main non-free" > /etc/apt/sources.list.d/backports.list
RUN apt-get update --allow-releaseinfo-change && apt-get install -t buster-backports -y libprotobuf-dev libprotobuf17 protobuf-compiler nlohmann-json3-dev g++
RUN apt-get install -y vim gdb cmake wget unzip git

WORKDIR /app/dist
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcpu.zip && unzip "libtorch-cxx11-abi-shared-with-deps-1.5.0+cpu.zip"

WORKDIR /app

COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc

FROM ubuntu:16.04

RUN apt-get update
# build toolchain.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install meson ninja-build build-essential
# Libraries for build.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-good1.0-dev
# Needed in docker because the container is at his minimum.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install software-properties-common
# Libraries for runtime.
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:nnstreamer/ppa
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install nnstreamer nnstreamer-dev nnstreamer-tensorflow-lite tensorflow-dev tensorflow-lite-dev

# ADD current directory (git clone dir) to the docker.
ADD . /buildroot
# Change current working directory to the git clone.
WORKDIR /buildroot
# Build
RUN mkdir -p build-docker
RUN meson build-docker
RUN ninja -C build-docker

RUN ./build-docker/rt_image_classification -h
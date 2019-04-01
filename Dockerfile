FROM ubuntu:16.04

RUN apt-get update
# build toolchain.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install meson ninja-build build-essential
# Libraries for build.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install libgtk-3-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-good1.0-dev
# Libraries for runtime.

# ADD current directory (git clone dir) to the docker.
ADD . /buildroot
# Change current working directory to the git clone.
WORKDIR /buildroot
# Build
RUN mkdir -p build-docker
RUN meson build-docker
RUN ninja -C build-docker
FROM ubuntu:16.04

RUN apt-get update
# build toolchain.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install meson ninja-build build-essential
# Libraries for build.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev
# Needed in docker because the container is at his minimum.
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install software-properties-common
# Libraries for runtime.
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:nnstreamer/ppa
RUN apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install nnstreamer nnstreamer-dev nnstreamer-tensorflow-lite tensorflow-dev tensorflow-lite-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install tensorflow-lite-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install libcairo2-dev
# For runtime
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install gstreamer1.0-plugins-good gstreamer1.0-plugins-base-apps gstreamer1.0-plugins-base

# ADD current directory (git clone dir) to the docker.
ADD . /buildroot
# Change current working directory to the git clone.
WORKDIR /buildroot
RUN make
# It expected to fail but at leat will loads all dynamic libraries.
RUN GST_DEBUG=3 ./rt_image_classification
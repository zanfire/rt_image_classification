# rt_image_classification

Demo application that show integration of Gstreamer and tensorflow-lite.

## Requirements

This demo application requires for compilation and runtime:
  - GStreamer (base and goods)
  - tensorflow-lite (use ppa ppa:nnstreamer/ppa because provides a nice library)
  - cairo
  - gcc
  - meson, ninja or make

Setup steps:

Add nnstreamer ppa for tensorflow-lite.

``
add-apt-repository ppa:nnstreamer/ppa
apt-get update
``

Min set for compile

``
sudo apt-get install meson ninja-build build-essential
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libcairo2-dev tensorflow-lite-dev
``

For runtime
``
sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-base-apps gstreamer1.0-plugins-base
``
 
See Dockerfile for an updated list.

## Building - meson

``
mkdir build
meson build
ninja -C build
``

## Building - make

``
make
``

## Building - docker (target ubuntu 16.04)

This will create a docker container with the build step and environments for Ubunutu 16.04

``
docker build --rm -t build .
``

## Run

This demo application will load a tensorflow lite file and use the video camera for image classification.
The model provided (mobilenet) is a slitly changed version of the one provided by tensorflow host.

### Run - options

``
Application Options:
  -d, --device=/dev/video0                                    device path
  -m, --model=mobilenet/mobilenet_v1_1.0_224_quant.tflite     model path
  -l, --label=mobilenet/labels.txt                            label path
  -t, --tensor                                                tensor name for overlay
  -c, --channel=0                                             tensor channel for overlay
``

### Run - examples

Uses webcam /dev/video1 and show the overlay for tensor's output *MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6* channel 2.

``
./rt_image_classification -d /dev/video1 -c 2 -t MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6 
``

With the shipped model you can access to the overlay of:

``
MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6
MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6
MobilenetV1/Logits/AvgPool_1a/AvgPool
MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd
MobilenetV1/Logits/SpatialSqueeze
MobilenetV1/Predictions/Reshape_1
``


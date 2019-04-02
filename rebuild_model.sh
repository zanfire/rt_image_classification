#/bin/bash -e

python3 $HOME/.local/lib/python3.6/site-packages/tensorflow/lite/python/tflite_convert.py --output_file=mobilenet_v1_1.0_224_quant.tflite \
--graph_def_file=mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant_frozen.pb \
--inference_type=QUANTIZED_UINT8 \
--input_shape="1,224, 224,3" \
--input_arrays=input \
--std_dev_value="128" \
--mean_value="127" \
--output_arrays=MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6,\
MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6,\
MobilenetV1/Logits/AvgPool_1a/AvgPool,\
MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd,\
MobilenetV1/Logits/SpatialSqueeze,\
MobilenetV1/Predictions/Reshape_1



#include "model.h"

// C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
// C
#include <gst/gst.h>
#include <limits.h>

#include "tensorflow/contrib/lite/optional_debug_tools.h"

// These dimensions need to match those the model was trained with.
constexpr int wanted_input_width = 224;
constexpr int wanted_input_height = 224;
constexpr int wanted_input_channels = 3;
constexpr float input_mean = 128.0f;
constexpr float input_std = 127.0f;
const std::string input_layer_name = "input";
const std::string output_layer_name = "softmax1";

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
void get_top_N(std::vector<float>& prediction, int num_results, float threshold, std::vector<std::pair<float, int> >* top_results) {
  //g_print("prediction size %d", prediction_size);
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;
  
  for (size_t i = 0; i < prediction.size(); i++) {
    float value = prediction[i];
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > (size_t)num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

// Preprocess the input image and feed the TFLite interpreter buffer for a float model.
void process_input_float_model(uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
  for (int y = 0; y < wanted_input_height; ++y) {
    float* out_row = buffer + (y * wanted_input_width * wanted_input_channels);
    for (int x = 0; x < wanted_input_width; ++x) {
      const int in_x = (y * image_width) / wanted_input_width;
      const int in_y = (x * image_height) / wanted_input_height;
      uint8_t* input_pixel =
          input + (in_y * image_width * image_channels) + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_input_channels);
      for (int c = 0; c < wanted_input_channels; ++c) {
        out_pixel[c] = (input_pixel[c] - input_mean) / input_std;
      }
    }
  }
}

// Preprocess the input image and feed the TFLite interpreter buffer for a quantized model.
void process_input_quant_model(
    uint8_t* input, uint8_t* output, int image_width, int image_height, int image_channels) {
  for (int y = 0; y < wanted_input_height; ++y) {
    uint8_t* out_row = output + (y * wanted_input_width * wanted_input_channels);
    for (int x = 0; x < wanted_input_width; ++x) {
      const int in_x = (y * image_width) / wanted_input_width;
      const int in_y = (x * image_height) / wanted_input_height;
      uint8_t* in_pixel = input + (in_y * image_width * image_channels) + (in_x * image_channels);
      uint8_t* out_pixel = out_row + (x * wanted_input_channels);
      for (int c = 0; c < wanted_input_channels; ++c) {
        out_pixel[c] = in_pixel[c];
      }
    }
  }
}


bool Model::load(char const* model, char const* label, char const* tensor_name, int channel) {
  model_path_ = model;
  label_path_ = label;
  if (tensor_name != nullptr) {
    tensor_name_ = tensor_name;
  }
  channel_ = channel;

  bool res1 = load_model(model);
  bool res2 = load_labels(label);
  bool res3 = (res1 && res2) ? activate() : false;
  g_print("Load model %s and labels %s, activated %s\n", res1 ? "OK" : "FAIL", res2 ? "OK" : "FAIL", res3 ? "OK" : "FAIL");
  return res1 && res2 && res3;
}

bool Model::load_model(char const* path) {
  model_ = tflite::FlatBufferModel::BuildFromFile(path);
  return model_.get() != nullptr;
}

bool Model::activate() {
  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  if (interpreter_.get() != nullptr) {
    // Resize input tensor.
    int input = interpreter_->inputs()[0];
    std::vector<int> sizes = {1, 224, 224, 3};
    interpreter_->ResizeInputTensor(input, sizes);
    // Allocate tensor buffers.
    interpreter_->AllocateTensors();
    tflite::PrintInterpreterState(interpreter_.get());

    int idx = 0;
    auto outs = interpreter_->outputs();
    for (auto& o : outs) {
      g_print("%d: Output %d %s\n", idx, o, interpreter_->GetOutputName(idx));
      idx++;
    }


    return true;
  }
  else {
    return false;
  }
}


bool Model::load_labels(char const* path) {
  std::ifstream input(path);
  
  if(!input) {
    GST_ERROR("Failed to open %s.", path);
    return false;
  }

  // Clear labels vector.
  labels_.clear();
  std::string line;
  while (std::getline(input, line)) {
    if(!line.empty()) {
      GST_DEBUG("Loading label %d %s\n", (int)(labels_.size() + 1), line.c_str());
      labels_.push_back(line);
    }
  }
  input.close();

  // Add padding
  const int padding = 16;
  while (labels_.size() % padding) {
    labels_.emplace_back();
  }
  return true;
}

std::string Model::get_label(){
  if (index_ >= 0 && index_ < (int)labels_.size()) {
    std::ostringstream out;
    out << labels_[index_] << " - ";
    out.precision(2);
    out << std::fixed << acc_;
    return out.str();
  }
  return "";
}

std::vector<uint8_t> Model::get_overlay(int* width) {
  std::lock_guard<std::mutex> guard(overlayMtx_);
  if (width != nullptr) *width = overlayFrameWidth_; 
  return overlayFrame_;
}


void Model::on_new_frame(guint8 * buffer, guint len) {
  if (buffer == nullptr) return;

  constexpr int image_width = 224;
  constexpr int image_height = 224;
  constexpr int image_channels = 4;

  int input = interpreter_->inputs()[0];
  TfLiteTensor *input_tensor = interpreter_->tensor(input);

  bool is_quantized = (input_tensor->type == kTfLiteUInt8);

  if (is_quantized) {
    uint8_t* out = interpreter_->typed_tensor<uint8_t>(input);
    process_input_quant_model(buffer, out, image_width, image_height, image_channels);
  } 
  else {
    float* out = interpreter_->typed_tensor<float>(input);
    process_input_float_model(buffer, out, image_width, image_height, image_channels);
  }

  if (interpreter_->Invoke() != kTfLiteOk) {
    GST_ERROR("Failed to invoke!");
  }

  // read output size from the output sensor
  auto outs = interpreter_->outputs();
  int reshapeIndex = 0;
  int intIndex = -1;
  for (size_t i = 0; i < outs.size(); i++) {
    if (!g_strcmp0(interpreter_->GetOutputName(i), "MobilenetV1/Predictions/Reshape_1")) {
      reshapeIndex = (int)i;
    }
    if (!g_strcmp0(interpreter_->GetOutputName(i), tensor_name_.c_str())) {
      intIndex = (int)i;
    }
  }

  auto output = get_tensor_output_2dim(reshapeIndex);
  std::vector<std::pair<float, int> > top_results;
  get_top_N(output, 5, 0.1, &top_results);
  // Set the label 
  for (size_t i = 0; i < top_results.size(); i++) {
    auto el = top_results[i];
    //g_print("Result %f %d - %s\n", el.first, el.second, labels_[el.second].c_str());
    index_ = el.second;
    acc_ = el.first;
  }

  // TODO: 
  if (intIndex >= 0) {
    std::lock_guard<std::mutex> guard(overlayMtx_);
    overlayFrame_ = get_tensor_output_mat_quant(intIndex, channel_, &overlayFrameWidth_);
  }

  return;
}


std::vector<float> Model::get_tensor_output_2dim(int idx) {
  std::vector<float> result;
  int input = interpreter_->inputs()[0];
  TfLiteTensor *input_tensor = interpreter_->tensor(input);
  bool is_quantized = false;
  if (input_tensor->type == kTfLiteUInt8) {
    is_quantized = true;
  }

    // Reshape get result
  const int output_tensor_index = interpreter_->outputs()[idx];
  TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_index);
  TfLiteIntArray* output_dims = output_tensor->dims;
  int output_size = output_dims->data[1];
  if (output_dims->size == 4) {
    output_size = output_dims->data[1] * output_dims->data[2] * output_dims->data[3];
  }
  //g_print("Output dims size %d - output size %d quant %s\n", output_dims->size, output_size, is_quantized ? "yes" : "false");

  if (is_quantized) {
    uint8_t* quantized_output = interpreter_->typed_output_tensor<uint8_t>(idx);
    int32_t zero_point = input_tensor->params.zero_point;
    float scale = input_tensor->params.scale;
    int step =  output_dims->size == 4 ? output_dims->data[3] : 1; 
    for (int i = 0; i < output_size; i += step) {
      result.push_back((quantized_output[i] - zero_point) * scale);
    }
    
  } else {
    float* output = interpreter_->typed_output_tensor<float>(0);
    for (int i = 0; i < output_size; i++) {
      result.push_back(output[i]);
    }
  }
  return result;
}

std::vector<uint8_t> Model::get_tensor_output_mat_quant(int idx, int channel, int* width) {
  std::vector<uint8_t> result;
  TfLiteTensor* tensor = interpreter_->tensor(interpreter_->outputs()[idx]);
  bool is_quantized = true;
  TfLiteIntArray* input_dims = tensor->dims;
  int intput_size = input_dims->data[1];
  if (input_dims->size != 4 ) {
    return result;
  }
  if (input_dims->size == 4 && (channel < 0 || channel > input_dims->data[3])) {
    return result;
  }
  
  intput_size = input_dims->data[1] * input_dims->data[2] * input_dims->data[3];
  //g_print("tensor dims size %d - %d %d %d %d\n", input_dims->size, input_dims->data[0], input_dims->data[1], input_dims->data[2], input_dims->data[3]);
  //g_print("tensor dims size %d - output size %d quant %s\n", input_dims->size, intput_size, is_quantized ? "yes" : "false");

  if (is_quantized) {
    uint8_t* quantized_input = interpreter_->typed_output_tensor<uint8_t>(idx);
    int step = input_dims->data[3]; 
    for (int i = channel; i < intput_size; i += step) {
      result.push_back(quantized_input[i]);
    }
  }
  if (width != nullptr) {
    *width = input_dims->data[1];
  }
  return result;
}

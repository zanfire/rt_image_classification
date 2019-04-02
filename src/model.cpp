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
constexpr float input_mean = 127.5f;
constexpr float input_std = 127.5f;
const std::string input_layer_name = "input";
const std::string output_layer_name = "softmax1";

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
void GetTopN(
    const float* prediction, const int prediction_size, const int num_results,
    const float threshold, std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;

  const long count = prediction_size;
  for (int i = 0; i < count; ++i) {
    const float value = prediction[i];
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
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
void ProcessInputWithFloatModel(uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
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
void ProcessInputWithQuantizedModel(
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


bool Model::load(char const* model, char const* label) {
  model_path_ = model;
  label_path_ = label;

  bool res1 = load_model(model);
  bool res2 = load_labels(label);
  g_print("Load model %s and labels %s\n", res1 ? "OK" : "FAIL", res1 ? "OK" : "FAIL");

  if (!res1 || !res2) {
    return false;
  }
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
    return true;
  }
  else {
    return false;
  }
}

bool Model::load_model(char const* path) {
  model_ = tflite::FlatBufferModel::BuildFromFile(path);
  return model_.get() != nullptr;
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
  /*const int padding = 16;
  while (lables_.size() % padding) {
    lablel_.emplace_back();
  }*/
  return true;
}

bool Model::update(guint8 * scores, guint len) {
  if (scores == nullptr) return false;
  if (len > labels_.size()) {
    return false;
  }
  guint8 max_score = 0;
  int old_index = index_;
  index_ = -1;
  // TODO: I don't like cast in this way.
  for (int i = 0; i < (int)len; i++) {
    if (scores[i] > 0 && scores[i] > max_score) {
      index_ = i;
      max_score = scores[i];
    }
  }
  acc_ = max_score / (float)UCHAR_MAX; // score / MAX_UCHAR
  GST_DEBUG("Update index %d -> %d\n", old_index, index_);
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


void Model::onNewFrame(guint8 * scores, guint len) {
  if (scores == nullptr) return false;
  
  g_print("New frames %d -> %d\n", old_index, index_);
  return true;
}


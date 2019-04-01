#include "model.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <gst/gst.h>

#include <limits.h>

bool Model::load(char const* model, char const* label) {
  model_path_ = model;
  label_path_ = label;

  bool res = loadLabels(label);
  return res;
}

bool Model::loadLabels(char const* path) {
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


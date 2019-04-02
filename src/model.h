#ifndef RT_IMG_CLASS_MODEL_LOADER_H__
#define RT_IMG_CLASS_MODEL_LOADER_H__

#include <glib.h>
#include <string>
#include <vector>

// Tensorflow library.
// TODO: I don't like to put so much in the headers ... reduce to the essential.
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"
#include "tensorflow/contrib/lite/string_util.h"

/**
 * @brief This class load and verify the model. 
 * 
 * 
 */
class Model {
private:
  std::string model_path_;
  std::string label_path_;
  std::vector<std::string> labels_;

  int index_;
  float acc_ = 0.0f;

  // tensorflow lite 
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

public:
  /**
   * @brief Construct a new Model Loader object
   * 
   * @param model 
   */
  Model() = default;
  ~Model() = default;

  bool load(char const* model, char const* label);

  void onNewFrame(guint8* buffer, guint size);

  /**
   * @brief 
   * 
   * @param score 
   * @param len 
   * @return true 
   * @return false 
   */
  bool update(guint8* score, guint len);
  
  /**
   * @brief Return current label and prob.
   * 
   * @return std::string 
   */
  std::string get_label();

private:
  bool load_model(char const* model);
  bool load_labels(char const* path);
};

#endif
#ifndef RT_IMG_CLASS_MODEL_LOADER_H__
#define RT_IMG_CLASS_MODEL_LOADER_H__

#include <glib.h>
#include <string>
#include <vector>
#include <mutex>

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
  std::string tensor_name_;
  int channel_ = 0;
  // Label index/
  int index_;
  float acc_ = 0.0f;

  // tensorflow lite 
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  // Data for render the overlay.
  std::mutex overlayMtx_;
  std::vector<uint8_t> overlayFrame_;
  int overlayFrameWidth_ = 0;

public:
  /**
   * @brief Construct a new Model Loader object
   * 
   * @param model 
   */
  Model() = default;
  ~Model() = default;

  bool load(char const* model, char const* label, char const* tensor_name, int channel);

  /**
   * @brief On new frame.
   * 
   * @param buffer 
   * @param size 
   */
  void on_new_frame(guint8* buffer, guint size);
  
  /**
   * @brief Return current label and prob.
   * 
   * @return std::string 
   */
  std::string get_label();

  /**
   * @brief Get the overlay frame.
   * 
   * @param width 
   * @return std::vector<uint8_t> 
   */
  std::vector<uint8_t> get_overlay( int* width);

private:
  bool load_model(char const* model);
  bool load_labels(char const* path);
  bool activate();
  /**
   * @brief Save the tensor output scaled to float.
   * 
   */
  std::vector<float> get_tensor_output_2dim(int idx);
  /**
   * Save the output matrix.
   * 
   * @remark not thread safe.
   * 
   * @param idx 
   * @param channel 
   * @returnstd::vector<uint8_t>>
   */
  std::vector<uint8_t> get_tensor_output_mat_quant(int idx, int channel, int* width);
};

#endif
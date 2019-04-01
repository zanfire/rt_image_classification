#ifndef RT_IMG_CLASS_MODEL_LOADER_H__
#define RT_IMG_CLASS_MODEL_LOADER_H__

#include <glib.h>
#include <string>
#include <vector>


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

public:
  /**
   * @brief Construct a new Model Loader object
   * 
   * @param model 
   */
  Model() = default;
  ~Model() = default;

  bool load(char const* model, char const* label);

  bool update(guint8* score, guint len);

  std::string get_label();

private:
  bool loadLabels(char const* path);
};

#endif
#ifndef RT_IMG_CLASS_APPLICATION_H__
#define RT_IMG_CLASS_APPLICATION_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <memory>

#include "model.h"

/**
 * @brief Application
 * 
 * Here we store all things that we need to hold for make the application run and do 
 * what we want...
 * 
 * - Run gstreamer pipeline
 * - Host callback from gstreamer.
 * - Retain the model and load the model.
 */
class Application {
public:
  // Main loop.
  std::unique_ptr<GMainLoop, decltype(&g_main_loop_unref)> mainloop_;
  // Bus watcher.
  std::unique_ptr<GstBus, decltype(&gst_object_unref)> bus_;
  guint bus_signal_id_message_ = 0;
  guint bus_signal_id_tensor_sink_new_data_ = 0;
  guint timer_id_ = 0;
  GstVideoInfo currentVideoInfo_;
  bool currentVideoInfoValid_ = false;
  // Pipeline.
  std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline_;
  // Model 
  Model model_;

public:
  /**
   * @brief Construct a new Application object (allocate the new main loop.)
   * 
   */
  Application() : mainloop_(g_main_loop_new(nullptr, false), &g_main_loop_unref),
                  bus_(nullptr, &gst_object_unref),
                  pipeline_(nullptr, &gst_object_unref) {};
  ~Application() = default;

  /**
   * @brief Setup the application and acquire needed resources (main loop and gst bus)
   * 
   * @return true if setup is correct otherwise false.
   */
  bool setup(char const* device, char const* model, char const* label, char const* tensor_name, int channel);

  /**
   * @brief Run the main loop.
   */
  void run();

  /**
   * @brief Signal main loop to quit.
   * 
   */
  void quit();
};

#endif
#ifndef RT_IMG_CLASS_APPLICATION_H__
#define RT_IMG_CLASS_APPLICATION_H__

#include <glib.h>
#include <gst/gst.h>

#include <memory>

#include "model.h"

class Application {
public:
  // Gtk main window/
  //GtkWidget* main_window = nullptr;
  // Main loop.
  std::unique_ptr<GMainLoop, decltype(&g_main_loop_unref)> mainloop_;
  // Bus watcher.
  std::unique_ptr<GstBus, decltype(&gst_object_unref)> bus_;
  guint bus_signal_id_message_ = 0;
  guint bus_signal_id_tensor_sink_new_data_ = 0;
  guint timer_id_ = 0;

  // Pipeline.
  std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline_;

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
  bool setup(char const* device, char const* model, char const* label);

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
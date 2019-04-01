#include "application.h"

//
// Callback from GstBus
//
static void onBusMessage(GstBus * bus, GstMessage * message, gpointer user_data) {
  Application* app = (Application*)user_data;
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS: {
      GST_ERROR("received eos message");
      app->quit();
      break;
    }
    case GST_MESSAGE_ERROR: {
      gchar* debug = nullptr;
      GError* error = nullptr;
      gst_message_parse_error(message, &error, &debug);
      char const* srcName = GST_OBJECT_NAME(message->src);
      GST_ERROR("Error (src: %s): %s - %s", srcName, error->message, debug);
      g_error_free(error);
      g_free(debug);
      
      app->quit();
      break;
    }
    default:
      break;
  }
}


static void onTensorSinkNewData(GstElement * element, GstBuffer * buffer, gpointer user_data) {
  Application* app = (Application*)user_data;

  int buffers = gst_buffer_n_memory(buffer);
  GST_DEBUG("onTensorSinkNewData num buffers %d.", buffers);
  for (int i = 0; i < buffers; i++) {
    GstMemory* mem = gst_buffer_peek_memory (buffer, i);
    GstMapInfo info;
    if (gst_memory_map (mem, &info, GST_MAP_READ)) {
      GST_DEBUG("Mapped memory %p size %d", info.data, (int) info.size);
      app->model_.update(info.data, (guint) info.size);
      gst_memory_unmap (mem, &info);
    }
  }
}

static gboolean onTimerCallback(gpointer user_data) {
  Application* app = (Application*)user_data;

  GstElement* overlay = gst_bin_get_by_name (GST_BIN (app->pipeline_.get()), "tensor_res");
  if (overlay) {
    std::string label = app->model_.get_label();
    g_object_set (overlay, "text", label.empty() ? " - " :  label.c_str(), nullptr);
    gst_object_unref(overlay);
    GST_DEBUG("Updating label %s\n", label.c_str());
  }
  return true; // return true mean contine.
}

bool Application::setup(char const* device, char const* model, char const* label) {

  model_.load(model, label);

  // TODO: cross-platform you can switch v4l2src to othe elemenet for supporting windows and macosx.
  char* pipeline = g_strdup_printf("v4l2src name=cam_src device=%s ! videoconvert ! videoscale ! "
      "video/x-raw,width=640,height=480,format=RGB ! tee name=t_raw "
      "t_raw. ! queue ! textoverlay name=tensor_res font-desc=Sans,24 ! "
      "videoconvert ! ximagesink name=img_tensor "
      "t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! tensor_converter ! "
      "tensor_filter framework=tensorflow-lite model=%s silent=false ! "
      "tensor_sink name=tensor_sink", device, model);


  g_print("Creating pipeline from string: %s\n", pipeline);

  pipeline_.reset(gst_parse_launch(pipeline, NULL));
  g_free(pipeline);

  if (pipeline_.get() == nullptr) {
    GST_ERROR("Failed to allocate pipeline.");
    return false;
  }
  // Get GstBus
  bus_.reset(gst_element_get_bus(pipeline_.get()));
  // Setup message bus signal.
  gst_bus_add_signal_watch (bus_.get());
  // TODO: rename here to be more clear.
  bus_signal_id_message_ = g_signal_connect(bus_.get(), "message", (GCallback) onBusMessage, this);
  // Tensor callback
  GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline_.get()), "tensor_sink");
  if (element != nullptr) {
    g_print("tensor sink found\n");
    bus_signal_id_tensor_sink_new_data_ = g_signal_connect(element, "new-data", (GCallback)onTensorSinkNewData, this);
    gst_object_unref (element);
  }

    /* timer to update result */
  timer_id_ = g_timeout_add (1000, onTimerCallback, this);

  return true;
}

void Application::run() {
  // start pipeline
  gst_element_set_state (pipeline_.get(), GST_STATE_PLAYING);

  /* set window title */
  //_set_window_title ("img_mixed", "Mixed");
  //_set_window_title ("img_origin", "Original");
  // start main loop
  g_main_loop_run(mainloop_.get());
  // TODO: Shutdown pipeline.
  gst_element_set_state (pipeline_.get(), GST_STATE_NULL);
}

void Application::quit() {
  g_main_loop_quit(mainloop_.get());
}
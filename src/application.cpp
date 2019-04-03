#include "application.h"
// Gstreamer
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>
// Cairo (for overlay)
#include <cairo.h>
#include <cairo-gobject.h>

#include <math.h>

//
// Callback from GstBus
//
static void on_bus_message(GstBus * bus, GstMessage * message, gpointer user_data) {
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
    case GST_MESSAGE_WARNING: {
      GST_ERROR("received warning message");
      app->quit();
      break;
    }
    default:
      break;
  }
}

//
// Call back from appsink with new frame for the model.
//
static GstFlowReturn onAppSinkNewData(GstElement * element, gpointer user_data) {
  Application* app = (Application*)user_data;
 
  GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK (element));
  GstBuffer* buffer = gst_sample_get_buffer (sample);

  int buffers = gst_buffer_n_memory(buffer);
  GST_DEBUG("onAppSinkNewData num buffers %d.", buffers);
  for (int i = 0; i < buffers; i++) {
    GstMemory* mem = gst_buffer_peek_memory (buffer, i);
    GstMapInfo info;
    if (gst_memory_map (mem, &info, GST_MAP_READ)) {
      GST_DEBUG("Mapped memory %p size %d", info.data, (int) info.size);
      app->model_.on_new_frame(info.data, (guint) info.size);
      gst_memory_unmap (mem, &info);
    }
  }
  return GST_FLOW_OK;
}

//
// Store the information from the caps that we are interested in. 
//
static void on_prepare_overlay(GstElement * overlay, GstCaps * caps, gpointer user_data) {
  Application* app = (Application*)user_data;
  app->currentVideoInfoValid_ = gst_video_info_from_caps (&app->currentVideoInfo_, caps);
}

//
// Draw text and image overlay in the cairooverlay element.
//
static void on_draw_overlay(GstElement * overlay, cairo_t * cr, guint64 timestamp, guint64 duration, gpointer user_data) {
  Application* app = (Application*)user_data;
  if (!app->currentVideoInfoValid_) {
    return;
  }
  int w = 0;
  // Get overlay frame and width from the model.
  // This is a copy.
  auto frame = app->model_.get_overlay(&w);
  int h = w;
  uint32_t* data = nullptr;

  //
  // Here we write the classification of the object.
  //
  cairo_scale(cr, 1.0, 1.0);
  cairo_set_source_rgba(cr, 0.3, 0.3, 0.3, 1.0);
  cairo_select_font_face(cr, "Monospace",
      CAIRO_FONT_SLANT_NORMAL,
      CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(cr, 24);
  cairo_move_to(cr, 20, 20);
  std::string label = app->model_.get_label();
  cairo_show_text(cr, label.empty() ? " - " :  label.c_str());


  // 
  // Here we render the overlay.
  //
  // Draw the overlay if we have a valid size.
  if ((w * h) > 0) {

    int output_width = GST_VIDEO_INFO_WIDTH(&app->currentVideoInfo_);
    //int output_height = GST_VIDEO_INFO_HEIGHT(&app->currentVideoInfo_);
    float scale = output_width / (float)w;

    data = (uint32_t*)malloc(w * h * sizeof(uint32_t));

    for (int i = 0; i < (w * h); i++) {
      // By default 0x10 on the alpha channel
      data[i] = 0x10000000;
      // I like read, write Alpha 0x30 and red.
      data[i] = 0x50000000 | (frame[i] << 16);
    }
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, w);
    // Create a cairo images from the 
    cairo_surface_t* image = cairo_image_surface_create_for_data((uint8_t*)data, CAIRO_FORMAT_ARGB32, w, h, stride);
    
    cairo_matrix_t matrix;
    cairo_matrix_init_rotate(&matrix, 90 * M_PI/180);
    cairo_matrix_scale(&matrix, 1.0, -1.0);
    cairo_set_matrix(cr, &matrix);

    cairo_translate(cr, 0, 0);
    cairo_scale(cr, scale, scale);
    cairo_set_source_surface (cr, image, 0, 0);
    cairo_paint (cr);
    cairo_surface_destroy (image);
  }

  if (data != nullptr) free(data);
}


bool Application::setup(char const* device, char const* model, char const* label, char const* tensor_name, int channel) {

  if (!model_.load(model, label, tensor_name, channel)) {
    return false;
  }

  // TODO: cross-platform you can switch v4l2src to othe elemenet for supporting windows and macosx.

  //
  // Setup the pipeline
  //
  // v4l2src -> videoconvert (RGBx) -> videoscale (to 720p if needed) -> crop (we use a square not a rectangle) -> tee (split) /
  //
  //    tee -> queue -> videoconvert  cairooverlay -> ximagesink (render window)
  //    |
  //    -> queue ->videoscale (224x224) -> appsink (app sink push frames to the model)
  //
  char* pipeline = g_strdup_printf("v4l2src device=%s ! videoconvert ! videoscale ! "
      "video/x-raw,width=1280,height=720,format=RGBx ! videocrop top=0 left=280 right=280 bottom=0 ! tee name=t_raw "
      "t_raw. ! queue ! videoconvert ! cairooverlay name=tensor_overlay ! "
      " ximagesink name=img_tensor "
      "t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,width=224,height=224 !"
      "appsink name=tensor_sink", device);

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
  g_signal_connect(bus_.get(), "message", (GCallback) on_bus_message, this);
  
  // Setup appsink to push new video frame to the model.
  GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline_.get()), "tensor_sink");
  if (element != nullptr) {
    g_object_set(element, "emit-signals", TRUE, "sync", FALSE, NULL);
    bus_signal_id_tensor_sink_new_data_ = g_signal_connect(element, "new-sample", (GCallback)onAppSinkNewData, this);
    gst_object_unref(element);
  }
  // Setup the overlay to draw overlay and text.
  element = gst_bin_get_by_name(GST_BIN(pipeline_.get()), "tensor_overlay");
  if (element != nullptr) {
    g_signal_connect(element, "draw", (GCallback)on_draw_overlay, this);
    g_signal_connect(element, "caps-changed", (GCallback)on_prepare_overlay, this);
    gst_object_unref(element);
  }
  return true;
}

void Application::run() {
  // set pipeline in start state.
  auto res = gst_element_set_state(pipeline_.get(), GST_STATE_PLAYING);
  if (res == GST_STATE_CHANGE_FAILURE) {
    g_print("Failed to start gstreamer pipeline. Run with GST_DEBUG=3");
    return;
  }
  // run main loop. It will block until main loop quits.
  g_main_loop_run(mainloop_.get());
  // set pipline in stop (NULL) state.
  gst_element_set_state(pipeline_.get(), GST_STATE_NULL);
}

void Application::quit() {
  g_main_loop_quit(mainloop_.get());
}
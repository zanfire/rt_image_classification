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
      app->model_.onNewFrame(info.data, (guint) info.size);
      gst_memory_unmap (mem, &info);
    }
  }
  return GST_FLOW_OK;
}

/* Store the information from the caps that we are interested in. */
static void onPrepareOverlay(GstElement * overlay, GstCaps * caps, gpointer user_data) {
  Application* app = (Application*)user_data;
  app->currentVideoInfoValid_ = gst_video_info_from_caps (&app->currentVideoInfo_, caps);
}

/* Draw the overlay. 
 * This function draws a cute "beating" heart. */
static void onDrawOverlay(GstElement * overlay, cairo_t * cr, guint64 timestamp, guint64 duration, gpointer user_data) {
  Application* app = (Application*)user_data;
  if (!app->currentVideoInfoValid_) {
    return;
  }

  /* FIXME: this assumes a pixel-aspect-ratio of 1/1 */
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
    cairo_scale(cr, scale, scale);

    data = (uint32_t*)malloc(w * h * sizeof(uint32_t));

    for (int i = 0; i < (w * h); i++) {
      // By default 0x10 on the alpha channel
      data[i] = 0x10000000;
      // I like read, write Alpha 0x30 and red.
      data[i] = 0x30000000 | (frame[i] << 16);
    }
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, w);
    // Create a cairo images from the 
    cairo_surface_t* image = cairo_image_surface_create_for_data((uint8_t*)data, CAIRO_FORMAT_ARGB32, w, h, stride);

    cairo_set_source_surface (cr, image, 0, 0);
    cairo_paint (cr);
    cairo_surface_destroy (image);
  }

  if (data != nullptr) free(data);
}

bool Application::setup(char const* device, char const* model, char const* label, char const* tensor_name, int channel) {

  model_.load(model, label, tensor_name, channel);

  // TODO: cross-platform you can switch v4l2src to othe elemenet for supporting windows and macosx.
  char* pipeline = g_strdup_printf("v4l2src device=%s ! videoconvert ! videoscale ! "
      "video/x-raw,width=1280,height=720,format=RGBx ! videocrop top=0 left=280 right=280 bottom=0 ! tee name=t_raw "
      "t_raw. ! queue ! videoconvert ! cairooverlay name=tensor_overlay ! "
      " ximagesink name=img_tensor "
      "t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,width=224,height=224 !"
      "appsink name=tensor_sink", device);
// cairooverlay for other overlay !!!!!!

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
    g_object_set(element, "emit-signals", TRUE, "sync", FALSE, NULL);
    bus_signal_id_tensor_sink_new_data_ = g_signal_connect(element, "new-sample", (GCallback)onAppSinkNewData, this);
    gst_object_unref(element);
  }

  element = gst_bin_get_by_name(GST_BIN(pipeline_.get()), "tensor_overlay");
  if (element != nullptr) {
    g_signal_connect(element, "draw", (GCallback)onDrawOverlay, this);
    g_signal_connect(element, "caps-changed", (GCallback)onPrepareOverlay, this);
    gst_object_unref(element);
  }
  return true;
}

void Application::run() {
  // set pipeline in start state.
  gst_element_set_state (pipeline_.get(), GST_STATE_PLAYING);
  // run main loop. It will block until main loop quits.
  g_main_loop_run(mainloop_.get());
  // set pipline in stop (NULL) state.
  gst_element_set_state (pipeline_.get(), GST_STATE_NULL);
}

void Application::quit() {
  g_main_loop_quit(mainloop_.get());
}
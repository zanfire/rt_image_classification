// Base headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Glib, GST headers
#include <glib.h>
#include <gst/gst.h>

#include "application.h"

// TODO: Memory managmenet.
static char* device = nullptr;
static char* model = nullptr;
static char* label = nullptr;
static char* tensor_name = nullptr;

static GOptionEntry entries[] =
{
  { "device", 'd', 0, G_OPTION_ARG_STRING, &device, "device path", "/dev/video0" },
  { "model", 'm', 0, G_OPTION_ARG_STRING, &model, "model path", "mobilenet/mobilenet_v1_1.0_224_quant.tflite" },
  { "label", 'l', 0, G_OPTION_ARG_STRING, &label, "label path", "mobilenet/labels.txt" },
  { "tensor", 't', 0, G_OPTION_ARG_STRING, &tensor_name, "tensor name for overlay", nullptr },
  { nullptr }
};

int main(int argc, char **argv) {
  GError* error = nullptr;
  GOptionContext* context = g_option_context_new("- RealTime image classification.");
  g_option_context_add_main_entries(context, entries, "rt_image_classification");
  // Add GST options.
  g_option_context_add_group(context, gst_init_get_option_group());
  // Parse incoming arguments.
  if (!g_option_context_parse (context, &argc, &argv, &error)) {
    return EXIT_FAILURE;
  }
  // Inits
  gst_init(&argc, &argv);

  if (device == nullptr) device = g_strdup("/dev/video0");
  if (model == nullptr) model = g_strdup("mobilenet/mobilenet_v1_1.0_224_quant.tflite");
  if (label == nullptr) label = g_strdup("mobilenet/labels.txt");

  g_print("Starting application with camera device %s\n", device);

  Application app;
  if (app.setup(device, model, label, tensor_name)) {
    app.run();
    return EXIT_SUCCESS;
  }
  else {
    return EXIT_FAILURE;
  }
}


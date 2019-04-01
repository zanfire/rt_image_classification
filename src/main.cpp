// Base headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Glib, GST and GTK headers
#include <glib.h>
#include <gst/gst.h>
#include <gtk/gtk.h>

// TODO: Memory managmenet.
char* device = NULL;

static GOptionEntry entries[] =
{
  { "device", 'd', 0, G_OPTION_ARG_STRING, &device, "device path", "/dev/video0" },

  { NULL }
};

int main(int argc, char **argv) {
  GError* error = NULL;
  GOptionContext* context = g_option_context_new("- RealTime image classification.");
  g_option_context_add_main_entries(context, entries, "rt_image_classification");
  // Add GTK and GST options.
  g_option_context_add_group(context, gtk_get_option_group(TRUE));
  g_option_context_add_group(context, gst_init_get_option_group());
  // Parse incoming arguments.
  if (!g_option_context_parse (context, &argc, &argv, &error)) {
    return EXIT_FAILURE;
  }
  // Inits
  gtk_init(&argc, &argv);
  gst_init(&argc, &argv);

  return EXIT_SUCCESS;
}


project('rt_image_classification', 'cpp', default_options : ['c_std=c11'], license: 'MIT', version: '0.0.0')

# Disable warning for GstH264Parser (unstable API).
add_global_arguments('-DGST_USE_UNSTABLE_API', language : 'c')

glibdep = dependency('glib-2.0')
gtkdep = dependency('gtk+-3.0')
gstdep = dependency('gstreamer-1.0')
gstappdep = dependency('gstreamer-app-1.0')
gstcodecdep = dependency('gstreamer-codecparsers-1.0')

sources = [ 'src/main.cpp']
deps = [glibdep, gtkdep, gstdep, gstappdep, gstcodecdep]

executable('rt_image_classification', sources, dependencies: deps)

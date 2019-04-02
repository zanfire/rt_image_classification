project('rt_image_classification', 'cpp', default_options : ['cpp_std=c++11'], license: 'MIT', version: '0.0.0')

# Get the compiler.
cc = meson.get_compiler('cpp')
# Look for dependencies.
glibdep = dependency('glib-2.0')
gstdep = dependency('gstreamer-1.0')
gstappdep = dependency('gstreamer-app-1.0')
gstvideo = dependency('gstreamer-video-1.0')
tensorflow = dependency('tensorflow-lite')
cairo = dependency('cairo')
pthread = dependency('threads')
# For recent meson version
dl = cc.find_library('dl', required : false)
# For old meson version
#dl = find_library('dl', required : false)

sources = [ 'src/main.cpp', 'src/application.cpp', 'src/model.cpp']
deps = [glibdep, gstdep, gstappdep, gstvideo, tensorflow, pthread, dl, cairo]

executable('rt_image_classification', sources, dependencies: deps)


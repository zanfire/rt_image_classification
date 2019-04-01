project('rt_image_classification', 'cpp', default_options : ['cpp_std=c++11'], license: 'MIT', version: '0.0.0')

glibdep = dependency('glib-2.0')
gstdep = dependency('gstreamer-1.0')
gstappdep = dependency('gstreamer-app-1.0')

sources = [ 'src/main.cpp', 'src/application.cpp', 'src/model.cpp']
deps = [glibdep, gstdep, gstappdep]

executable('rt_image_classification', sources, dependencies: deps)


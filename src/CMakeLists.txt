cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

make_module(pyigl_proto
  BINDING_SOURCES matrix_add.cpp test_binding.cpp)

target_link_libraries(pyigl_proto PRIVATE igl::core igl::opengl_glfw)
target_compile_options(pyigl_proto PRIVATE "-fopenmp-simd")
target_compile_options(pyigl_proto PRIVATE "-mavx")
target_compile_options(pyigl_proto PRIVATE "-mfma")

#find_package(PythonLibs REQUIRED)
#target_link_libraries(pyigl_proto PRIVATE ${PYTHON_LIBRARIES})
#set_target_properties(pyigl_proto PROPERTIES LINK_FLAGS "-Wl,--no-undefined")


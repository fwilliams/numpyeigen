cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libigl/shared/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("${CMAKE_MODULE_PATH}")
# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_NANOGUI          "Use Nanogui menu"   OFF)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_PNG              "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)
include(${CMAKE_CURRENT_SOURCE_DIR}/pybind22Tools.cmake)

set (SRC_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

configure_file(${SRC_DIRECTORY}/binding_dsl.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

make_module(pyigl_proto ${SRC_DIRECTORY}/example_binding.cpp
  BINDING_SOURCES ${SRC_DIRECTORY}/matrix_add.cpp)
target_link_libraries(pyigl_proto PRIVATE igl::core igl::opengl_glfw)
target_compile_options(pyigl_proto PRIVATE "-fopenmp-simd")
target_compile_options(pyigl_proto PRIVATE "-mavx")
target_compile_options(pyigl_proto PRIVATE "-mfma")

#find_package(PythonLibs REQUIRED)
#target_link_libraries(pyigl_proto PRIVATE ${PYTHON_LIBRARIES})
#set_target_properties(pyigl_proto PROPERTIES LINK_FLAGS "-Wl,--no-undefined")


enable_testing()
add_subdirectory(tests)

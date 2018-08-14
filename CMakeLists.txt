cmake_minimum_required(VERSION 2.8.12)
project(numpyeigen)

add_subdirectory(external/pybind11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(numpyeigenTools)

add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE external/eigen)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

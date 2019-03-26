cmake_minimum_required(VERSION 3.2)
project(NumpyEigen VERSION 0.1.0)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(numpyeigen)

# Configuration options
set(NPE_PYTHON_EXECUTABLE    "" CACHE STRING "Path to the Python interpreter")
set(NPE_PYTHON_VERSION       "" CACHE STRING "Request a specific version of Python")

option(NPE_WITH_EIGEN        "Whether to use the bundled version of Eigen" OFF)
option(NPE_BUILD_TESTS       "Build NumpyEigen tests"                      OFF)

# Check if we're using NumpyEigen from another project via add_subdirectory
set(NPE_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(NPE_MASTER_PROJECT ON)
endif()

if (NPE_MASTER_PROJECT)
  set(NPE_BUILD_TESTS ON)
else()
  set(NPE_BUILD_TESTS OFF)
endif()

if(${NPE_BUILD_TESTS})
  enable_testing()
  add_subdirectory(tests)
endif()

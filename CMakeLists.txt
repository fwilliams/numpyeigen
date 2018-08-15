cmake_minimum_required(VERSION 3.2)
project(NumpyEigen VERSION 0.1.0)

# Check if we're using NumpyEigen from another project via add_subdirectory
set(NPE_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(NPE_MASTER_PROJECT ON)
endif()

if (NPE_MASTER_PROJECT)
  option(NPE_BUILD_TESTS "Build NumpyEigen tests"                              ON)
  option(NPE_WITH_EIGEN  "Use eigen submodule when building NumpyEigen module" ON)
else()
  option(NPE_BUILD_TESTS "Build NumpyEigen tests"                              OFF)
  option(NPE_WITH_EIGEN  "Use eigen submodule when building NumpyEigen module" OFF)
endif()


if(${NPE_BUILD_TESTS})
  enable_testing()
  add_subdirectory(tests)
endif()

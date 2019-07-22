# Include this file to set up NumpyEigen with your CMake project
#
#
# Setting the following variables before including this file will affect behavior:
#
# NPE_PYTHON_EXECUTABLE
#     Set the Python interpreter which is in turn used to find the correct
#     library and header files to link against.
# NPE_PYTHON_VERSION
#     Request a specific version of Python. NOTE: If NPE_PYTHON_EXECUTABLE is set
#     then we ignore this variable and issue a warning.
# NPE_WITH_EIGEN
#     Build NumpyEigen modules with the bundled version of Eigen. We include an
#     up-to-date version of Eigen for convenience, set this to use it.
# NPE_ROOT_DIR
#     The root directory of the NumpyEigen repository. If you're including the
#     original version of this file in the NumpyEigen repository, this will
#     be set automatically.
#


cmake_minimum_required(VERSION 3.2)

# Utility options to enable various vectorization optimizations in Eigen
set(NPE_EXTRA_CXX_FLAGS "")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPILER_SUPPORT_OPENMP)
if(COMPILER_SUPPORT_OPENMP)
  option(NPE_OPENMP  "Enable/Disable OpenMP" OFF)
endif()


option(NPE_SSE2   "Enable/Disable SSE2" OFF)
option(NPE_SSE3   "Enable/Disable SSE3" OFF)
option(NPE_SSSE3  "Enable/Disable SSSE3" OFF)
option(NPE_SSE4_1 "Enable/Disable SSE4.1" OFF)
option(NPE_SSE4_2 "Enable/Disable SSE4.2" OFF)
option(NPE_AVX    "Enable/Disable AVX" OFF)
option(NPE_AVX2   "Enable/Disable AVX" OFF)
option(NPE_AVX512 "Enable/Disable AVX" OFF)


if(NPE_OPENMP)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-fopenmp")
    message(STATUS "Enabling OpenMP")
endif()

if(NPE_SSE2)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse2")
  message(STATUS "Enabling SSE2 for NumpyEigen modules")
endif()

if(NPE_SSE3)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse3")
  message(STATUS "Enabling SSE3 for NumpyEigen modules")
endif()

if(NPE_SSSE3)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mssse3")
  message(STATUS "Enabling SSSE3 for NumpyEigen modules")
endif()

if(NPE_SSE4_1)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse4.1")
  message(STATUS "Enabling SSE4.1 for NumpyEigen modules")
endif()

if (NPE_SSE4_2)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse4.2")
  message(STATUS "Enabling SSE4.2 for NumpyEigen modules")
endif()

if(NPE_AVX)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mavx")
  message(STATUS "Enabling AVX for NumpyEigen modules")
endif()


if(NPE_AVX2)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mavx2")
  message(STATUS "Enabling AVX2 for NumpyEigen modules")
endif()

if(NPE_AVX512)
  set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mavx512f" "-DEIGEN_ENABLE_AVX512")
  message(STATUS "Enabling AVX512 for NumpyEigen modules")
endif()



# If you haven't set NPE_ROOT_DIR, then we're going to try and set it
if(NOT NPE_ROOT_DIR)
  set(NPE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
  if (NOT EXISTS ${NPE_ROOT_DIR}/src/npe.h)
    message(FATAL_ERROR "Could not find NumpyEigen. Set NPE_ROOT_DIR to point to the root directory of NumpyEigen.")
  endif()
  set(NPE_ROOT_DIR ${NPE_ROOT_DIR} PARENT_SCOPE)
# If you have set NPE_ROOT_DIR, then we'lll check that its valid
elseif(NOT EXISTS ${NPE_ROOT_DIR}/src/npe.h)
  message(FATAL_ERROR "NPE_ROOT_DIR (${NPE_ROOT_DIR}) does not point to a valid root directory for NumpyEigen.")
endif()




# If you specified NPE_PYTHON_EXECUTABLE, then we use this as the python interpreter
# which is in turn used to find the correct headers and link libraries.
# If NPE_PYTHON_EXECUTABLE is unset, then we just use whatever findPythonInterp turns up
#
# If you specified NPE_PYTHON_VERSION, then we will try and find a Python interpreter with
# that version. IF NPE_PYTHON_EXECUTABLE is set, then the version string is ignored
# and we use the interpreter specified by NPE_PYTHON_EXECUTABLE
if (NPE_PYTHON_EXECUTABLE)
  if (NPE_PYTHON_VERSION)
    message(WARNING "You set both NPE_PYTHON_EXECUTABLE and NPE_PYTHON_VERSION. NPE_PYTHON_EXECUTABLE will take precedence and the versions may not match.")
  endif()
  set(PYTHON_EXECUTABLE ${NPE_PYTHON_EXECUTABLE})
elseif(NPE_PYTHON_VERSION)
  set(PYBIND11_PYTHON_VERSION "${NPE_PYTHON_VERSION}")
endif()
# Include the bundled pybind11

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE "python")
endif()


# Get the path to the Python header files
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
  OUTPUT_VARIABLE NPE_PYTHON_INCLUDE_DIR)

if(NOT NPE_PYTHON_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Unable to find python include dir")
endif()

# Get the path to the Numpy header files
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
  OUTPUT_VARIABLE NPE_NUMPY_INCLUDE_DIR)

if(NOT NPE_NUMPY_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Unable to find numpy include dir")
endif()

if(UNIX AND NOT APPLE)
  LIST(APPEND NPE_EXTRA_CXX_FLAGS "-fPIC")
endif()


add_library(npe "${NPE_ROOT_DIR}/src/npe_typedefs.cpp")
target_include_directories(npe PUBLIC "${NPE_ROOT_DIR}/src" ${NPE_PYTHON_INCLUDE_DIR} ${NPE_NUMPY_INCLUDE_DIR})
target_compile_options(npe PUBLIC ${NPE_EXTRA_CXX_FLAGS})

# This needs to happen after the execute_process commands above
# The execute_process commands set variables which are ordinariely set by
# FindPython.cmake. If they are unset, pybindd11 will call FindPython and will
# use the default python in /usr/bin which breaks conda environments.
include(numpyeigenDependencies)

if(TARGET Eigen3::Eigen)
  # If an imported target already exists, use it
  target_link_libraries(npe PUBLIC Eigen3::Eigen)
else()
  if(NPE_WITH_EIGEN)
    target_include_directories(npe PUBLIC ${NPE_WITH_EIGEN})
  else()
    target_include_directories(npe PUBLIC ${NUMPYEIGEN_EXTERNAL}/eigen)
  endif()
endif()

target_link_libraries(npe PUBLIC pybind11::module)

# We require C++ 14 for auto return types
set_target_properties(npe PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)


# Create a Python module from the NumpyEigen src1.cpp, ..., srcN.cpp
#
# npe_add_module(
#     BINDING_SOURCES src1.cpp src2.cpp ... srcN.cpp
#     [MODULE | SHARED]
#     [EXClUDE_FROM_ALL]
#     [THIN_LTO])
#
# MODULE or SHARED specifies the type of library to build. The default is MODULE. These are
# the same as for add_library (https://cmake.org/cmake/help/v3.0/command/add_library.html)
#
# EXCLUDE_FROM_ALL is the same as for add_library
# (https://cmake.org/cmake/help/v3.0/command/add_library.html)
#
# THIN_LTO enables link time optimizations for some compilers (see for example
# https://clang.llvm.org/docs/ThinLTO.html)
#
# NOTE:This function assumes the variable NPE_ROOT_DIR is set to the root directory of NumpyEigen.
#
function(npe_add_module target_name)
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO DEBUG_TRACE)
  set(multiValueArgs BINDING_SOURCES)
  cmake_parse_arguments(npe_add_module "${options}" "" "${multiValueArgs}" ${ARGN})

  # If you included this file, then PYTHON_EXECUTABLE should always be set
  if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "No Python interpreter is defined. We expect the variable PYTHON_EXECUTABLE to be set!")
  endif()


  # Directory containing NumpyEigen source code
  set(NPE_SRC_DIR "${NPE_ROOT_DIR}/src")

  # NumpyEigen uses the C preprocessor for parsing. Here we find a valid command to invoke the C preprocessor
  if (MSVC)
    set(C_PREPROCESSOR_CMD_FLAGS "/E")
  else()
    set(C_PREPROCESSOR_CMD_FLAGS "-w -E")
  endif()

  # For each binding source file add a "target" which runs the NumpyEigen compiler when the binding code changes
  foreach(binding_source ${npe_add_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)

    set(bound_function_source_filename "${CMAKE_CURRENT_BINARY_DIR}/${name}.out.cpp")
    set(debug_trace_arg "")
    if (npe_add_module_DEBUG_TRACE)
      set(debug_trace_arg "--debug-trace")
    endif()

    add_custom_command(OUTPUT ${bound_function_source_filename}
      DEPENDS ${binding_source} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
      COMMAND ${PYTHON_EXECUTABLE} ${NPE_SRC_DIR}/codegen_function.py ${binding_source} ${CMAKE_CXX_COMPILER} -o ${bound_function_source_filename} ${debug_trace_arg} --c-preprocessor-args ${C_PREPROCESSOR_CMD_FLAGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    list(APPEND function_sources "${bound_function_source_filename}")
  endforeach()

  # Add a target which generates the C++ code for the whole output module
  set(module_source_filename ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.module.cpp)
  add_custom_command(OUTPUT ${module_source_filename}
    DEPENDS ${npe_add_module_BINDING_SOURCES} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
    COMMAND ${PYTHON_EXECUTABLE} ${NPE_SRC_DIR}/codegen_module.py -o ${module_source_filename} -m ${target_name} -f ${npe_add_module_BINDING_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Register the generated C++ code for the module as a pybind11 module so it gets compiled
  add_library(${target_name} MODULE ${module_source_filename} ${function_sources})
  target_link_libraries(${target_name} PUBLIC npe)
  set_target_properties(${target_name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
endfunction()


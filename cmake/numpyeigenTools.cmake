set(NPE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

function(npe_add_module target_name)
  find_program(CPP_PROGRAM, cpp)

  if (MSVC)
    set(C_PREPROCESSOR_CMD "${CMAKE_CXX_COMPILER} /E")
  else()
    find_program(cpp, FIND_CPP)
    if (NOT ${FIND_CPP})
      error("Failed to find C preprocessor. This is needed to generate bindings!")
    endif()
    set(C_PREPROCESSOR_CMD "cpp")
  endif()

  execute_process(COMMAND python -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
    OUTPUT_VARIABLE NP_INCLUDE_DIR)
  execute_process(
    COMMAND
      python -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)

  message("Current source dir ${CMAKE_CURRENT_SOURCE_DIR}")
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  set(multiValueArgs TARGET_SOURCES BINDING_SOURCES)
  cmake_parse_arguments(make_module "${options}" "" "${multiValueArgs}" ${ARGN})

  foreach(binding_source ${make_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)

    set(bound_function_source_filename "${CMAKE_CURRENT_BINARY_DIR}/${name}.out.cpp")
    add_custom_command(OUTPUT ${bound_function_source_filename}
      DEPENDS ${binding_source} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
      COMMAND python ${NPE_SRC_DIR}/codegen_function.py ${binding_source} ${C_PREPROCESSOR_CMD} -o ${bound_function_source_filename}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    list(APPEND function_sources "${bound_function_source_filename}")
  endforeach()

  set(module_source_filename ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.module.cpp)
  add_custom_command(OUTPUT ${module_source_filename}
    DEPENDS ${make_module_BINDING_SOURCES} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
    COMMAND python ${NPE_SRC_DIR}/codegen_module.py -o ${module_source_filename} -m ${target_name} -f ${make_module_BINDING_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  pybind11_add_module(${target_name} SHARED ${module_source_filename} ${NPE_SRC_DIR}/npe_typedefs.cpp ${function_sources})
  target_include_directories(${target_name} PRIVATE ${NPE_SRC_DIR} ${PYTHON_INCLUDE_DIR} ${NP_INCLUDE_DIR})

  # python ${make_module_BINDING_SOURCE}
endfunction()


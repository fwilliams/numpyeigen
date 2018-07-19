add_subdirectory(pybind11)

set (PYBIND22_SRC_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Empty list of binding file names
function(bind_function module_name filename)
  get_filename_component(name ${filename} NAME_WE)
  get_filename_component(dir ${filename} DIRECTORY)
  set(generator_target run_binding_generator_${module_name}_${name})
  add_custom_target(${generator_target}
    DEPENDS ${filename}
    COMMAND python ${SRC_DIRECTORY}/binding_dsl.py ${filename} -o ${dir}/${name}.out.cpp)
  add_dependencies(${module_name} ${generator_target})
endfunction()

function(make_module target_name target_cpp)
  # TODO: target_cpp is a hack for now
  message("Target name: ${target_name}")
  message("Target cpp: ${target_cpp}")
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  set(multiValueArgs TARGET_SOURCES BINDING_SOURCES)
  cmake_parse_arguments(make_module "${options}" "" "${multiValueArgs}" ${ARGN})

  pybind11_add_module(${target_name} SHARED ${PYBIND22_SRC_DIRECTORY}/binding_typedefs.cpp ${target_cpp} ${make_module_TARGET_SOURCES})

  foreach(target_source ${make_module_TARGET_SOURCES})
    message("TSRC: ${target_source}")
  endforeach()

  foreach(binding_source ${make_module_BINDING_SOURCES})
    bind_function(${target_name} ${binding_source})
  endforeach()

endfunction()


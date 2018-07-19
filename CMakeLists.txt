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


set (SRC_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

configure_file(${SRC_DIRECTORY}/binding_dsl.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

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

add_subdirectory(pybind11)
pybind11_add_module(pyigl_proto SHARED src/matrix_add.out.cpp src/binding_typedefs.cpp src/example_binding.cpp)
target_link_libraries(pyigl_proto PRIVATE igl::core igl::opengl_glfw)
#target_compile_options(pyigl_proto PRIVATE "-fopenmp-simd")
#target_compile_options(pyigl_proto PRIVATE "-mavx")
#target_compile_options(pyigl_proto PRIVATE "-mfma")
bind_function(pyigl_proto ${SRC_DIRECTORY}/matrix_add.cpp)

#find_package(PythonLibs REQUIRED)
#target_link_libraries(pyigl_proto PRIVATE ${PYTHON_LIBRARIES})
#set_target_properties(pyigl_proto PROPERTIES LINK_FLAGS "-Wl,--no-undefined")


enable_testing()
add_subdirectory(tests)

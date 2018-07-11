#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <igl/cotmatrix.h>
#include <iostream>
#include <tuple>
#include <cstdint>

#include "binding_typedefs.h"

namespace py = pybind11;
using namespace std;


PYBIND11_MODULE(pyigl_proto, m) {
    m.doc() = R"pbdoc(
        Playground for prototyping libIGL functions
        -----------------------

        .. currentmodule:: pyigl_proto

        .. autosummary::
           :toctree: _generate

           array_info
           cotmatrx
    )pbdoc";

#include "test_binding.out.cpp"
}

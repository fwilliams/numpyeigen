#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <tuple>

#include "binding_typedefs.h"
#include "binding_utils.h"

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
  #include "matrix_add.out.cpp"

    m.def("test_return", [](pybind11::array_t<double> a, pybind11::array_t<double> b) {
      typedef Eigen::Matrix<
          double,
          Eigen::Dynamic,
          Eigen::Dynamic,
          Eigen::RowMajor> Matrix_Type;
      Eigen::Map<Matrix_Type, Eigen::Aligned> A((double*)a.data(), a.shape()[0], a.shape()[1]);
      Eigen::Map<Matrix_Type, Eigen::Aligned> B((double*)a.data(), b.shape()[0], b.shape()[1]);

      Eigen::MatrixXd C = A + B;
      Eigen::MatrixXd D = A - B;
      return std::make_tuple(C, D);
    });
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

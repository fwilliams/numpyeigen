#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


PYBIND11_MODULE(numpyeigen_helpers, m) {
  m.doc() = R"pbdoc(
            Test Module
            -----------------------

            .. currentmodule:: test_module
            )pbdoc";

  m.def("mutate_copy", [](pybind11::array_t<float> v) {
    float* v_data = (float*) v.data();
    v_data[0] = 2.0;
    return v.shape()[0] + v.shape()[1];
  });
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

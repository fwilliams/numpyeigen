#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <sparse_array.h>

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

  m.def("sparse_return", [](npe::sparse_array sp) {
    std::string fmt = sp.getformat();

    std::cout << sp.indices().shape()[0] << " " << sp.data().shape()[0] << " " << fmt << std::endl;
    std::cout << sp.data().dtype().type() << std::endl;

    return sp;
  });
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

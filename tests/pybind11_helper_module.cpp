#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <npe_sparse_array.h>
#include <npe_typedefs.h>

PYBIND11_MODULE(numpyeigen_helpers, m) {
  m.doc() = R"pbdoc(
            Test Module
            -----------------------

            .. currentmodule:: test_module
            )pbdoc";

  m.def("mutate_copy", [](pybind11::array_t<float> v) {
    float* v_data = (float*) v.data();
    v_data[0] = 2.0;
    return v;
  });

  m.def("sparse_return", [](npe::sparse_array sp) {
    return sp;
  });

  m.def("sparse_mutate_copy", [](Eigen::SparseMatrix<double>& mat) {
    mat.coeffRef(0, 0) = 2.0;
    return std::make_tuple(mat.rows(), mat.cols());
  });

  m.def("return_dense_copy", [](pybind11::array_t<float> arr){
    return arr;
  }, pybind11::return_value_policy::copy);

  m.def("return_sparse_copy", [](int size) {
    Eigen::SparseMatrix<double> ret(size, size);
    ret.setIdentity();
    return ret;
  }, pybind11::return_value_policy::copy);

  m.def("test_dtype_cast", [](pybind11::dtype t) {
    pybind11::print(t);
  });
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

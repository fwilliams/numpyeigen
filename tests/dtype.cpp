#include <npe.h>

npe_function(test_dtype)

npe_arg(a, dense_float)
npe_default_arg(dtype, npe::dtype, "double")

npe_begin_code()

Eigen::MatrixXd ret_d;
Eigen::MatrixXf ret_f;
switch(dtype.type()) {
  case npe::type_f32:
    return npe::move(ret_f);
  case npe::type_f64:
    return npe::move(ret_d);
  default:
    throw pybind11::type_error("Only float32 and float64 dtypes are supported");
}

npe_end_code()

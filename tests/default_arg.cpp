
#include <iostream>
#include <tuple>
#include <numpyeigen_utils.h>

npe_function("default_arg")
npe_arg("a", "dense_f64")
npe_default_arg("b", "std::string", "std::string(\"abc\")")
npe_default_arg("c", "dense_f64", "dense_f32", "pybind11::array_t<double>()")
npe_begin_code()

a(0, 0) = 2.0;

return std::make_tuple(b + std::string("def"), NPE_MOVE_DENSE_MAP(c), NPE_MOVE_DENSE_MAP(a));

npe_end_code()

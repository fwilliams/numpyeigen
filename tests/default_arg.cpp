
#include <iostream>
#include <tuple>
#include <numpyeigen_utils.h>

npe_function("default_arg")
npe_arg("a", "type_f64")
npe_default_arg("b", "std::string", "std::string(\"abc\")")
npe_default_arg("c", "type_f64", "type_f32", "pybind11::array_t<double>()")
npe_begin_code()

std::cout << "mutate_matrix()" << std::endl;

npe::Map_a A((npe::Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
A(0, 0) = 2.0;

return std::make_tuple(b + std::string("def"), c, NPE_MOVE_DENSE_MAP(A));

npe_end_code()

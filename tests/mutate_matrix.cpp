
#include <iostream>
#include <numpyeigen_utils.h>

npe_function("mutate_matrix")
npe_arg("a", "type_f64")
npe_begin_code()

// a - py::array

std::cout << "mutate_matrix()" << std::endl;

npe::Map_a A((npe::Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);

A(0, 0) = 2.0;

//return A.rows() + A.cols();
return NPE_MOVE_DENSE_MAP(A);

npe_end_code()

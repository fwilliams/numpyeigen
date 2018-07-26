#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include "numpyeigen_utils.h"

npe_function("mutate_sparse_matrix")
npe_arg("a", "sparse_f64", "sparse_f32")
npe_begin_code()

using namespace std;

npe::Map_a A = a.as_eigen<npe::Matrix_a>();

A.coeffRef(0, 0) = 2.0;

return NPE_MOVE_SPARSE(A);

npe_end_code()

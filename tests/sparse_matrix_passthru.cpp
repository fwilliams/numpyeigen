#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include "numpyeigen_utils.h"

npe_function("sparse_matrix_passthru")
npe_arg("a", "sparse_f64", "sparse_f32")
npe_arg("b", "matches(a)")
npe_begin_code()

using namespace std;

npe::Map_a A = a.as_eigen<npe::Matrix_a>();
npe::Map_b B = b.as_eigen<npe::Matrix_b>();

npe::Matrix_a ret1 = A + B;

return NPE_MOVE_SPARSE(A);

npe_end_code()



#include <tuple>
#include <Eigen/Core>
#include "numpyeigen_utils.h"

npe_function("sparse_matrix_add")
npe_arg("a", "sparse_f64")
npe_arg("b", "matches(a)")
npe_begin_code()

using namespace std;

npe::Map_a A = a.as_eigen<npe::Matrix_a>();
npe::Map_b B = b.as_eigen<npe::Matrix_b>();

npe::Matrix_a ret1 = A + B;

return std::make_tuple(ret1.rows(), ret1.cols());

npe_end_code()



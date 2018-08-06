#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

npe_function(sparse_matrix_add)
npe_arg(a, sparse_f64, sparse_f32)
npe_arg(b, matches(a))
npe_begin_code()

npe_Matrix_a ret1 = a + b;

return NPE_MOVE_SPARSE(ret1);

npe_end_code()



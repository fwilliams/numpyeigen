#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

npe_function(sparse_matrix_passthru)
npe_arg(a, sparse_f64, sparse_f32)
npe_arg(b, matches(a))
npe_begin_code()

// This addition should have no effect which is what we test.
npe_Matrix_a ret1 = a + b;

return NPE_MOVE_SPARSE(a);

npe_end_code()



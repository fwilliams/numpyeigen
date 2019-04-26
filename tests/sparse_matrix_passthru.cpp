#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

npe_function(sparse_matrix_passthru)
npe_arg(a, sparse_double, sparse_float)
npe_arg(b, npe_matches(a))
npe_begin_code()

// This addition should have no effect which is what we test.
npe_Matrix_a ret1 = a + b;

return npe::move(a);

npe_end_code()



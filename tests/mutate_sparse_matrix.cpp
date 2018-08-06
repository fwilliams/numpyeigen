#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

npe_function(mutate_sparse_matrix)
npe_arg(a, sparse_f64, sparse_f32)
npe_begin_code()

a.coeffRef(0, 0) = 2.0;

return NPE_MOVE_SPARSE(a);

npe_end_code()

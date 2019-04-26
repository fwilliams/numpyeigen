#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

npe_function(mutate_sparse_matrix)
npe_arg(a, sparse_double, sparse_float)
npe_begin_code()

a.coeffRef(0, 0) = 2.0;

return npe::move(a);

npe_end_code()

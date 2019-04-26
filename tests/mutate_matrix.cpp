#include <iostream>
#include <Eigen/Core>
#include <npe.h>

npe_function(mutate_matrix)
npe_arg(a, dense_double, dense_float)
npe_begin_code()

a(0, 0) = 2.0;

return npe::move(a);

npe_end_code()

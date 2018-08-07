#include <iostream>
#include <Eigen/Core>
#include <npe.h>

npe_function(mutate_matrix)
npe_arg(a, dense_f64, dense_f32)
npe_begin_code()

std::cout << "mutate_matrix()" << std::endl;

a(0, 0) = 2.0;

return NPE_MOVE_DENSE_MAP(a);

npe_end_code()

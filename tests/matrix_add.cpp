#include <tuple>
#include <Eigen/Core>
#include <npe_utils.h>

npe_function("matrix_add")
npe_arg("a", "dense_f64", "dense_f32")
npe_arg("b", "matches(a)")
npe_begin_code()

npe_Matrix_a ret1 = a + b;

return NPE_MOVE_DENSE(ret1);

npe_end_code()



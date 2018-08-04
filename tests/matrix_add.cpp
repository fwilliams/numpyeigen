#include <tuple>
#include <Eigen/Core>
#include "numpyeigen_utils.h"

npe_function("matrix_add")
npe_arg("a", "type_f64", "type_f32")
npe_arg("b", "matches(a)")
npe_begin_code()

Matrix_a ret1 = a + b;

return NPE_MOVE_DENSE(ret1);

npe_end_code()



#include <tuple>
#include <Eigen/Core>
#include "numpyeigen_utils.h"

npe_function("matrix_add")
npe_arg("a", "type_f64", "type_f32")
npe_arg("b", "matches(a)")
npe_begin_code()

using namespace std;

npe::Map_a A((npe::Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
npe::Map_b B((npe::Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

npe::Matrix_a ret1 = A + B;

return NPE_MOVE_DENSE(ret1);

npe_end_code()



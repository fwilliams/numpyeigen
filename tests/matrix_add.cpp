#include <tuple>
#include <Eigen/Core>
#include "numpyeigen_utils.h"

npe_function("matrix_add")
npe_arg("a", "type_f64")
npe_arg("b", "matches(a)")
npe_begin_code()

using namespace std;

typedef NPE_PY_TYPE_a::Scalar Scalar_a;
typedef NPE_PY_TYPE_a::Eigen_Type Matrix_a;
typedef NPE_PY_TYPE_a::Map_Type Map_a;

typedef NPE_PY_TYPE_b::Scalar Scalar_b;
typedef NPE_PY_TYPE_b::Eigen_Type Matrix_b;
typedef NPE_PY_TYPE_b::Map_Type Map_b;

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
Map_b B((Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

Matrix_a ret1 = A + B;

return ret1.rows() + ret1.rows();

npe_end_code()



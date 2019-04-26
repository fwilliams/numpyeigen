#include <tuple>
#include <Eigen/Core>
#include <npe.h>

npe_function(matrix_add)
npe_arg(b, npe_matches(a))
npe_arg(a, dense_double, dense_float)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a + b;

    return npe::move(ret1);

npe_end_code()




npe_function(matrix_add2)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_matches(a))
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a + b;

    return npe::move(ret1);

npe_end_code()




npe_function(matrix_add3)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_matches(a))
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a + b;

    return npe::move(ret1);

npe_end_code()




npe_function(matrix_add4)
npe_arg(a, dense_double, dense_float)
npe_arg(b, dense_double, dense_float)
npe_doc(R"(Add two matrices)")
npe_begin_code()

    npe_Matrix_a ret1 = a + b.template cast<npe_Scalar_a>();

    return npe::move(ret1);

npe_end_code()

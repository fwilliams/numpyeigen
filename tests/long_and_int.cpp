#include <tuple>
#include <Eigen/Core>
#include <npe.h>

npe_function(intlonglong)
npe_arg(b, dense_int, dense_longlong)
npe_arg(a, dense_int, dense_longlong)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()




npe_function(intlong)
npe_arg(b, dense_int, dense_long)
npe_arg(a, dense_int, dense_long)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()




npe_function(longlonglong)
npe_arg(b, dense_longlong, dense_long)
npe_arg(a, dense_longlong, dense_long)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()

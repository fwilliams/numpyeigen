#include <tuple>
#include <Eigen/Core>
#include <npe.h>

npe_function(int32int64)
npe_arg(b, dense_int32, dense_int64)
npe_arg(a, dense_int32, dense_int64)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()

npe_function(int64int32)
npe_arg(b, dense_int64, dense_int32)
npe_arg(a, dense_int64, dense_int32)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()



npe_function(uint32uint64)
npe_arg(b, dense_uint32, dense_uint64)
npe_arg(a, dense_uint32, dense_uint64)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()

npe_function(uint64uint32)
npe_arg(b, dense_uint64, dense_uint32)
npe_arg(a, dense_uint64, dense_uint32)
npe_doc(R"(Add two matrices of the same type)")
npe_begin_code()

    npe_Matrix_a ret1 = a;

    return npe::move(ret1);

npe_end_code()



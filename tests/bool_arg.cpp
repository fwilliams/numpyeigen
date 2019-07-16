#include <tuple>
#include <Eigen/Core>
#include <npe.h>

npe_function(bool_array)
npe_arg(a, dense_bool)
npe_arg(b, npe_matches(a))
npe_doc(R"(Add two boolean matrices)")
npe_begin_code()

    npe_Matrix_a ret1 = a + b;

    return npe::move(ret1);

npe_end_code()


npe_function(bool_array_sparse)
npe_arg(a, sparse_bool)
npe_arg(b, npe_matches(a))
npe_doc(R"(Add two boolean matrices)")
npe_begin_code()

    npe_Matrix_a ret1 = a + b;

    return npe::move(ret1);

npe_end_code()




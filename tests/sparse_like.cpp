#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

npe_function(sparse_like_1)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_sparse_like(a))
npe_begin_code()

npe_Matrix_b ret1 = a * b;

return npe::move(ret1);
npe_end_code()



npe_function(sparse_like_2)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_sparse_like(a))
npe_arg(c, npe_sparse_like(a))
npe_begin_code()

npe_Matrix_b ret1 = b;

return npe::move(ret1);
npe_end_code()



npe_function(sparse_like_3)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_sparse_like(a))
npe_arg(c, npe_sparse_like(b))
npe_begin_code()

npe_Matrix_b ret1 = b;

return npe::move(ret1);
npe_end_code()


npe_function(sparse_like_4)
npe_arg(a, sparse_double, sparse_float)
npe_arg(b, npe_sparse_like(a))
npe_arg(c, npe_sparse_like(b))
npe_begin_code()

npe_Matrix_b ret1 = b;

return npe::move(ret1);
npe_end_code()

#include <iostream>
#include <Eigen/Core>
#include <npe.h>

npe_function(default_matches_1)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_matches(a))
npe_default_arg(c, npe_matches(b))
npe_begin_code()

a(0, 0) = 2.0;

if (c.rows() != 0 || c.cols() != 0) {
  c(0, 0) = 2.0;
}

return std::make_tuple(npe::move(a), npe::move(b), npe::move(c));

npe_end_code()




npe_function(default_matches_2)
npe_arg(a, dense_double, dense_float)
npe_arg(b, npe_matches(a))
npe_default_arg(c, npe_matches(a))
npe_arg(d, dense_long, dense_int)
npe_default_arg(e, npe_matches(d))
npe_default_arg(f, npe_matches(e))
npe_begin_code()

d(0, 0) = 2;

if (e.rows() != 0 || e.cols() != 0) {
  e(0, 0) = 2;
}

if (f.rows() != 0 || f.cols() != 0) {
  f(0, 0) = 2;
}

return std::make_tuple(npe::move(d), npe::move(e), npe::move(f));

npe_end_code()



npe_function(default_matches_3)
npe_arg(a, sparse_double, sparse_float)
npe_arg(b, npe_matches(a))
npe_default_arg(c, npe_matches(a))
npe_arg(d, dense_long, dense_int)
npe_default_arg(e, npe_matches(d))
npe_default_arg(f, npe_matches(e))
npe_begin_code()

d(0, 0) = 2;

if (e.rows() != 0 || e.cols() != 0) {
  e(0, 0) = 2;
}

if (f.rows() != 0 || f.cols() != 0) {
  f(0, 0) = 2;
}
if (c.rows() != 0 || c.cols() != 0) {
  return std::make_tuple(npe::move(a), npe::move(b), npe::move(c));
} else {
  return std::make_tuple(npe::move(d), npe::move(e), npe::move(f));
}

npe_end_code()

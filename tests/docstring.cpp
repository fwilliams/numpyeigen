#include <iostream>
#include <Eigen/Core>
#include <npe.h>

npe_function(docstring)
npe_arg(a, dense_f64, dense_f32)

// This is a comment

// and another
npe_doc(R"(This is
a multi-line
documentation


string...
)")

// and another comment
// and enen one more
npe_begin_code()

a(0, 0) = 2.0;

return npe::move(a);

npe_end_code()

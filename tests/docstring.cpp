#include <iostream>
#include <Eigen/Core>
#include <npe.h>

// TODO: Multiline literal docstrings are broken with the C preprocessor and older compilers
const char* docstring = R"(This is
a multi-line
documentation
string...
)";

npe_function(docstring)
npe_arg(a, dense_double, dense_float)

// This is a comment

// and another

npe_doc(docstring)

// and another comment
// and even one more
npe_begin_code()

auto npe_arg = [](int x) { return x + 1; };
npe_arg(5);
a(0, 0) = 2.0;

return npe::move(a);

npe_end_code()

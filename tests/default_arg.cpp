#include <iostream>
#include <tuple>
#include <npe.h>

npe_function(default_arg)
npe_arg(a, dense_f64)
npe_default_arg(b, std::string, std::string("abc"))
npe_default_arg(c, dense_f64, dense_f32, pybind11::array_t<double>())
npe_default_arg(doubleit, bool, false)
npe_begin_code()

a(0, 0) = 2.0;

std::string retstr = "";
if(doubleit) {
  retstr = b;
}
return std::make_tuple(retstr + b + std::string("def"), NPE_MOVE_DENSE_MAP(c), NPE_MOVE_DENSE_MAP(a));

npe_end_code()

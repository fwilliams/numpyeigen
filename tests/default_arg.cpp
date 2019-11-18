#include <iostream>
#include <tuple>
#include <npe.h>

npe_function(default_arg)
npe_arg(a, dense_double)
npe_default_arg(b, std::string, std::string("abc"))
npe_default_arg(c, dense_double, dense_float, pybind11::array_t<double>())
npe_default_arg(doubleit, bool, false)
npe_begin_code()

a(0, 0) = 2.0;

std::string retstr = "";
if(doubleit) {
  retstr = b;
}
return std::make_tuple(retstr + b + std::string("def"), npe::move(c), npe::move(a));

npe_end_code()


npe_function(default_arg2)
npe_arg(a, dense_double)
npe_default_arg(b, std::string, std::string("abc"))
npe_default_arg(c, dense_double, dense_float, pybind11::array_t<double>())
npe_default_arg(doubleit, bool, false)
npe_begin_code()

Eigen::MatrixXi ret(1, 2);
ret(0, 0) = a.rows();
ret(0, 1) = a.cols();

return npe::move(ret);

npe_end_code()

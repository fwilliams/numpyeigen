#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/eigen.h>

#define NPE_MOVE(eig_var) \
  pybind11::reinterpret_steal<pybind11::object>(pybind11::detail::eigen_encapsulate<pybind11::detail::EigenProps<decltype(eig_var)>>(new decltype(eig_var)(std::move(eig_var))))

#endif // BINDING_UTILS_H

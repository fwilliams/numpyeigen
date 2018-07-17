#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/eigen.h>

#define MOVE_TO_NP(Type, eig_var) \
  pybind11::reinterpret_steal<pybind11::object>(pybind11::detail::eigen_encapsulate<pybind11::detail::EigenProps<Type>>(new Type(std::move(eig_var))))

#endif // BINDING_UTILS_H

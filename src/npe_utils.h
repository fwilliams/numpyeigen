#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/eigen.h>
#include <npe_sparse_array.h>
#include <npe_dense_array.h>
#include <type_traits>


namespace npe {
template <typename T> using is_eigen_dense_map = pybind11::detail::is_eigen_dense_map<T>;
template <typename T> using is_eigen_dense = pybind11::detail::is_eigen_dense_plain<T>;
template <typename T> using is_eigen_sparse = pybind11::detail::is_eigen_sparse<T>;
template <typename T> using is_eigen_other = pybind11::detail::is_eigen_other<T>;
template <typename T> using is_eigen = pybind11::detail::is_template_base_of<Eigen::EigenBase, T>;
template <typename T> using is_not_eigen = pybind11::detail::negation<is_eigen<T>>;

template <typename T, typename std::enable_if<is_eigen_dense<T>::value>::type* = nullptr>
pybind11::object move(T& ret, bool squeeze = true) {
  return pybind11::reinterpret_steal<pybind11::object>(
        npe::detail::eigen_encapsulate_dense<pybind11::detail::EigenProps<T>>(new T(std::move(ret)), squeeze));
}

template <typename T, typename std::enable_if<is_eigen_dense_map<T>::value>::type* = nullptr>
pybind11::object move(T& ret, bool squeeze = true) {
  return pybind11::reinterpret_steal<pybind11::object>(
        npe::detail::eigen_encapsulate_dense_map<pybind11::detail::EigenProps<T>>(new T(std::move(ret)), squeeze));
}

template <typename T, typename std::enable_if<is_eigen_sparse<T>::value>::type* = nullptr>
pybind11::object move(T& ret, bool squeeze = true /* unused for sparse types */) {
  return pybind11::reinterpret_steal<pybind11::object>(pybind11::detail::eigen_encapsulate_sparse<T>(new T(std::move(ret))));
}

template <typename T, typename std::enable_if<is_eigen_other<T>::value>::type* = nullptr>
pybind11::object move(T&, bool squeeze = true /* unused for other types */) {
  static_assert(!is_eigen_other<T>::value, "Called npe::move on invalid type. Can only call it on Eigen::Matrix, Eigen::SparseMatrix, or Eigen::Map<T>s where T is a Matrix or SparseMatrix.");
  return pybind11::object();
}

template <typename T, typename std::enable_if<is_not_eigen<T>::value>::type* = nullptr>
pybind11::object move(T, bool squeeze = true /*unused for other types */) {
  static_assert(!is_not_eigen<T>::value, "Called npe::move on invalid type. Can only call it on Eigen::Matrix, Eigen types.");
  return pybind11::object();
}


}
#endif // BINDING_UTILS_H

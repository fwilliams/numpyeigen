#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/eigen.h>
#include <npe_sparse_array.h>
#include <npe_dense_array.h>
#include <type_traits>

// TODO: These macros suck and let's make one general template function to make these work
#define NPE_MOVE_DENSE(eig_var) \
  pybind11::reinterpret_steal<pybind11::object>( \
  pybind11::detail::eigen_encapsulate<pybind11::detail::EigenProps<decltype(eig_var)>>( \
  new decltype(eig_var)(std::move(eig_var))))

#define NPE_MOVE_DENSE_MAP(eig_var) \
  pybind11::reinterpret_steal<pybind11::object>( \
  pybind11::detail::eigen_encapsulate_dense_map<pybind11::detail::EigenProps<decltype(eig_var)>>( \
  new decltype(eig_var)(std::move(eig_var))))

#define NPE_MOVE_SPARSE(eig_var) \
  pybind11::reinterpret_steal<pybind11::object>( \
  pybind11::detail::eigen_encapsulate_sparse<decltype(eig_var)>( \
  new decltype(eig_var)(std::move(eig_var))))


//namespace npe {
//template <typename T> using is_eigen_dense_mutable_map =
//  pybind11::detail::all_of<
//    pybind11::detail::is_template_base_of<Eigen::DenseBase, T>,
//    std::is_base_of<Eigen::MapBase<T, Eigen::WriteAccessors>, T>>;
//template <typename T> using is_eigen_dense =
//  pybind11::detail::any_of<
//    pybind11::detail::is_eigen_dense_map<T>,
//    is_eigen_dense_mutable_map<T>,
//    pybind11::detail::is_eigen_dense_plain<T>>;

//template <typename T, typename = typename std::enable_if<is_eigen_dense<T>::value>::type>
//pybind11::object npe_move(T& ret) {
//  return pybind11::reinterpret_steal<pybind11::object>(
//        pybind11::detail::eigen_encapsulate<pybind11::detail::EigenProps<T>(new T(std::move(ret))));
//}


//template <typename T, typename = typename std::enable_if<pybind11::detail::is_eigen_sparse<T>::value>::type>
//pybind11::object npe_move(T& ret) {
//  return  pybind11::reinterpret_steal<pybind11::object>(pybind11::detail:: eigen_encapsulate_sparse(new T(std::move(ret))));
//}
//}
#endif // BINDING_UTILS_H

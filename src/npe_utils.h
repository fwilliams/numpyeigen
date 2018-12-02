#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
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


namespace detail {

template <typename T>
struct maybe_none : public T {
  bool is_none = false;
};

} // namespace detail
} // namespace npe


namespace pybind11 {
namespace detail {

template <>
struct type_caster<npe::detail::maybe_none<pybind11::array>> {
public:
  PYBIND11_TYPE_CASTER(npe::detail::maybe_none<pybind11::array>, _("numpy.array | None"));

  bool load(handle src, bool arg) {
    if (!src.is_none()) {
      pybind11::detail::type_caster<pybind11::array> subcaster;

      bool ret = subcaster.load(src, arg);
      if (!ret) {
        return ret;
      }
      static_cast<pybind11::array&>(value) = *subcaster;
      value.is_none = false;
    } else {
      value.is_none = true;
    }

    return true;
  }

  static handle cast(npe::detail::maybe_none<pybind11::array> mn, return_value_policy policy, handle parent) {
    if (mn.is_none) {
      return pybind11::none();
    } else {
      return pybind11::detail::type_caster<pybind11::array>::cast(static_cast<pybind11::array&>(mn), policy, parent);
    }
  }
};


template <>
struct type_caster<npe::detail::maybe_none<npe::sparse_array>> {
public:
  PYBIND11_TYPE_CASTER(npe::detail::maybe_none<npe::sparse_array>, _("scipy.csr_matrix | scipy.csc_matrix | None"));

  bool load(handle src, bool arg) {
    if (!src.is_none()) {
      pybind11::detail::type_caster<npe::sparse_array> subcaster;

      bool ret = subcaster.load(src, arg);
      if (!ret) {
        return ret;
      }
      static_cast<npe::sparse_array&>(value) = *subcaster;
      value.is_none = false;
    } else {
      object sparse_module = module::import("scipy.sparse");
      object matrix_type = sparse_module.attr("csr_matrix");
      static_cast<npe::sparse_array&>(value) = matrix_type(0);
      static_cast<npe::sparse_array&>(value).hack_update_flags();
      value.is_none = true;
    }

    return true;
  }

  static handle cast(npe::detail::maybe_none<npe::sparse_array> mn, return_value_policy policy, handle parent) {
    if (mn.is_none) {
      return pybind11::none();
    } else {
      return pybind11::detail::type_caster<npe::sparse_array>::cast(static_cast<npe::sparse_array&>(mn), policy, parent);
    }
  }
};

} // namespace detail
} // namespace pybind11

#endif // BINDING_UTILS_H

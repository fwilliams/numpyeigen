#ifndef NPE_SPARSE_ARRAY_H
#define NPE_SPARSE_ARRAY_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <tuple>
#include <pybind11/eigen.h>
#include <numpy/arrayobject.h>

#include <Eigen/Core>

namespace npe {

// TODO: Cache attributes so we don't do expensive lookups more than once
struct sparse_array : pybind11::object {
  PYBIND11_OBJECT(sparse_array, pybind11::object, PyType_Check)

  sparse_array() = default;

  pybind11::dtype dtype() const {
    auto atr = this->attr("dtype");
    return pybind11::reinterpret_borrow<pybind11::dtype>(atr);
  }

  pybind11::array data() const {
    auto atr = this->attr("data");
    return pybind11::reinterpret_borrow<pybind11::array>(atr);
  }

  pybind11::array indices() const {
    auto atr = this->attr("indices");
    return pybind11::reinterpret_borrow<pybind11::array>(atr);
  }

  pybind11::array indptr() const {
    auto atr = this->attr("indptr");
    return pybind11::reinterpret_borrow<pybind11::array>(atr);
  }

  std::pair<ssize_t, ssize_t> shape() const {
    return this->attr("shape").cast<std::pair<ssize_t, ssize_t>>();
  }

  std::string getformat() const {
    return this->attr("getformat")().cast<std::string>();
  }

  bool row_major() const {
    return getformat() == "csr";
  }

  bool col_major() const {
    return !row_major();
  }

  int flags() const {
    return _flags;
  }

  int nnz() const {
    return this->attr("nnz").cast<int>();
  }

  template <typename T>
  Eigen::MappedSparseMatrix<typename T::Scalar, T::Options, typename T::StorageIndex> as_eigen() {
    std::pair<ssize_t, ssize_t> shape = this->shape();
    pybind11::array data = this->data();
    pybind11::array indices = this->indices();
    pybind11::array indptr = this->indptr();

    // TODO: Use Eigen::Map in post 3.3
    typedef Eigen::MappedSparseMatrix<typename T::Scalar, T::Options, typename T::StorageIndex> RetMap;
    return RetMap(shape.first, shape.second, this->nnz(),
                  (typename T::StorageIndex*) indices.data(),
                  (typename T::StorageIndex*) indptr.data(),
                  (typename T::Scalar*)data.data());
  }

private:
  friend class pybind11::detail::type_caster<npe::sparse_array>;
  int _flags = 0;
};

}

namespace pybind11 {
namespace detail {


template <>
struct type_caster<npe::sparse_array> {
public:
  PYBIND11_TYPE_CASTER(npe::sparse_array, _("scipy.sparse.csr_matrix | scipy.sparse.csc_matrix"));

  /**
   * Conversion part 1 (Python->C++): convert a PyObject into a npe::sparse_array
   * instance or return false upon failure. The second argument
   * indicates whether implicit conversions should be applied.
   */
  bool load(handle src, bool) {
    if (!src) {
      return false;
    }

    // TODO: attribute lookups are kind of slow and I would like to avoid them so lets cache the results
    // TODO: Check if attribute lookup succeeded
    try {
      std::string fmt = pybind11::getattr(src, "getformat")().cast<std::string>();
      pybind11::getattr(src, "data").cast<pybind11::array>();
      pybind11::getattr(src, "indices").cast<pybind11::array>();
      pybind11::getattr(src, "indptr").cast<pybind11::array>();
      if (fmt != "csr" && fmt != "csc") {
        return false;
      }

      value = pybind11::reinterpret_borrow<npe::sparse_array>(src);
      value._flags |= (value.col_major() ? NPY_ARRAY_F_CONTIGUOUS : NPY_ARRAY_C_CONTIGUOUS);

    } catch (pybind11::cast_error) {
      return false;
    }

    return true;
  }

  /**
   * Conversion part 2 (C++ -> Python): convert a npe::sparse_array instance into
   * a Python object. The second and third arguments are used to
   * indicate the return value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  static handle cast(npe::sparse_array src, return_value_policy /* policy */, handle /* parent */) {
    return src.release();
  }

};

}
}

#endif // NPE_SPARSE_ARRAY_H

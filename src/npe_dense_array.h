#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <type_traits>

#ifndef DENSE_ARRAY_H
#define DENSE_ARRAY_H



namespace npe {

template <typename T> using is_eigen_dense_map = pybind11::detail::is_eigen_dense_map<T>;
template <typename T> using is_eigen_dense = pybind11::detail::is_eigen_dense_plain<T>;

namespace detail {

// Casts an Eigen type to numpy array.  If given a base, the numpy array references the src data,
// otherwise it'll make a copy.  writeable lets you turn off the writeable flag for the array.
// squeeze will squeeze any 1 sized dimensions
template <typename props>
pybind11::handle eigen_array_cast(typename props::Type const &src,
                                  pybind11::handle base = pybind11::handle(),
                                  bool writeable = true,
                                  bool squeeze = true) {
    constexpr ssize_t elem_size = sizeof(typename props::Scalar);
    pybind11::array a;
    if (props::vector) {
        a = pybind11::array({ src.size() }, { elem_size * src.innerStride() }, src.data(), base);
    } else {
        a = pybind11::array({ src.rows(), src.cols() }, { elem_size * src.rowStride(), elem_size * src.colStride() },
                  src.data(), base);
    }

    if (!writeable) {
        pybind11::detail::array_proxy(a.ptr())->flags &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    }

    if (squeeze) {
      a = a.squeeze();
    }

    return a.release();
}

// Takes an lvalue ref to some Eigen type and a (python) base object, creating a numpy array that
// reference the Eigen object's data with `base` as the python-registered base class (if omitted,
// the base will be set to None, and lifetime management is up to the caller).  The numpy array is
// non-writeable if the given type is const.
template <typename props, typename Type>
pybind11::handle eigen_ref_array(Type &src, pybind11::handle parent = pybind11::none(), bool squeeze = true) {
    // none here is to get past array's should-we-copy detection, which currently always
    // copies when there is no base.  Setting the base to None should be harmless.
    return eigen_array_cast<props>(src, parent, !std::is_const<Type>::value /*writable*/, squeeze);
}


// Takes a pointer to some dense, plain Eigen type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename props, typename Type, typename = std::enable_if<is_eigen_dense_map<Type>::value, void>>
pybind11::handle eigen_encapsulate_dense_map(Type *src, bool squeeze = true) {
    pybind11::capsule base(src, [](void *o) {
      delete static_cast<Type *>(o);
    });
    return eigen_ref_array<props>(*src, base, squeeze);
}



// Takes a pointer to some dense, plain Eigen type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename props, typename Type, typename = std::enable_if<is_eigen_dense<Type>::value, void>>
pybind11::handle eigen_encapsulate_dense(Type *src, bool squeeze = true) {
    pybind11::capsule base(src, [](void *o) {
      delete static_cast<Type *>(o);
    });
    return eigen_ref_array<props>(*src, base, squeeze);
}

}
}

#endif // DENSE_ARRAY_H

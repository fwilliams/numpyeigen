#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#ifndef DENSE_ARRAY_H
#define DENSE_ARRAY_H

namespace pybind11 {
namespace detail {

// Takes a pointer to some dense, plain Eigen type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename props, typename Type, typename = enable_if_t<is_eigen_dense_map<Type>::value>>
handle eigen_encapsulate_dense_map(Type *src) {
    capsule base(src, [](void *o) {
      delete static_cast<Type *>(o);
    });
    return eigen_ref_array<props>(*src, base);
}

}
}
#endif // DENSE_ARRAY_H

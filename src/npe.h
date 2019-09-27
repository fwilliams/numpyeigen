#ifndef NPE_H
#define NPE_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <numpy/npy_common.h>

#include <npe_utils.h>
#include <npe_typedefs.h>
#include <npe_sparse_array.h>
#include <npe_dtype.h>

#include <Eigen/Sparse>
#include <Eigen/Core>

static_assert(EIGEN_WORLD_VERSION >= 3 || EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION >= 2,
              "NumpyEigen does not support verions prior to 3.2. Please update Eigen to at least version 3.2");


// We define a bunch of stuff to make your ide work with npe_* syntax and produce valid types.
// When we actually generate the bindings all of this is gone
#ifndef __NPE_FOR_REAL__

namespace npe {
  namespace detail {

    typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_double;
    typedef Eigen::Matrix<npy_float, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_float;

    typedef Eigen::Matrix<npy_byte, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_byte;
    typedef Eigen::Matrix<npy_short, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_short;
    typedef Eigen::Matrix<npy_int, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_int;
    typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_long;
    typedef Eigen::Matrix<npy_longlong, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_longlong;

    typedef Eigen::Matrix<npy_ubyte, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_ubyte;
    typedef Eigen::Matrix<npy_ushort, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_ushort;
    typedef Eigen::Matrix<npy_uint, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_uint;
    typedef Eigen::Matrix<npy_ulong, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_ulong;
    typedef Eigen::Matrix<npy_ulonglong, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_ulonglong;


    typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_double;
    typedef Eigen::Matrix<npy_float, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_float;

    typedef Eigen::Matrix<npy_byte, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_byte;
    typedef Eigen::Matrix<npy_short, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_short;
    typedef Eigen::Matrix<npy_int, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_int;
    typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_long;
    typedef Eigen::Matrix<npy_longlong, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_longlong;

    typedef Eigen::Matrix<npy_ubyte, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_ubyte;
    typedef Eigen::Matrix<npy_ushort, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_ushort;
    typedef Eigen::Matrix<npy_uint, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_uint;
    typedef Eigen::Matrix<npy_ulong, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_ulong;
    typedef Eigen::Matrix<npy_ulonglong, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_ulonglong;



    typedef Eigen::Map<npe_Matrix_dense_double, Eigen::Aligned> sparse_double;
    typedef Eigen::Map<npe_Matrix_dense_float, Eigen::Aligned> sparse_float;

    typedef Eigen::Map<npe_Matrix_dense_byte, Eigen::Aligned> sparse_byte;
    typedef Eigen::Map<npe_Matrix_dense_short, Eigen::Aligned> sparse_short;
    typedef Eigen::Map<npe_Matrix_dense_int, Eigen::Aligned> sparse_int;
    typedef Eigen::Map<npe_Matrix_dense_long, Eigen::Aligned> sparse_long;
    typedef Eigen::Map<npe_Matrix_dense_longlong, Eigen::Aligned> sparse_longlong;

    typedef Eigen::Map<npe_Matrix_dense_ubyte, Eigen::Aligned> sparse_ubyte;
    typedef Eigen::Map<npe_Matrix_dense_ushort, Eigen::Aligned> sparse_ushort;
    typedef Eigen::Map<npe_Matrix_dense_uint, Eigen::Aligned> sparse_uint;
    typedef Eigen::Map<npe_Matrix_dense_ulong, Eigen::Aligned> sparse_ulong;
    typedef Eigen::Map<npe_Matrix_dense_ulonglong, Eigen::Aligned> sparse_ulonglong;



    typedef Eigen::Map<npe_Matrix_sparse_double, Eigen::Aligned> dense_double;
    typedef Eigen::Map<npe_Matrix_sparse_float, Eigen::Aligned> dense_float;
    typedef Eigen::Map<npe_Matrix_sparse_byte, Eigen::Aligned> dense_byte;
    typedef Eigen::Map<npe_Matrix_sparse_short, Eigen::Aligned> dense_short;
    typedef Eigen::Map<npe_Matrix_sparse_int, Eigen::Aligned> dense_int;
    typedef Eigen::Map<npe_Matrix_sparse_long, Eigen::Aligned> dense_long;
    typedef Eigen::Map<npe_Matrix_sparse_long, Eigen::Aligned> dense_longlong;
    typedef Eigen::Map<npe_Matrix_sparse_ubyte, Eigen::Aligned> dense_ubyte;
    typedef Eigen::Map<npe_Matrix_sparse_ushort, Eigen::Aligned> dense_ushort;
    typedef Eigen::Map<npe_Matrix_sparse_uint, Eigen::Aligned> dense_uint;
    typedef Eigen::Map<npe_Matrix_sparse_ulong, Eigen::Aligned> dense_ulong;
    typedef Eigen::Map<npe_Matrix_sparse_ulonglong, Eigen::Aligned> dense_ulonglong;
  }

}


#define __NPE_GET_FIRST(arg, ...) arg
#define __NPE_MATRIX_TYPE(arg, ...) npe_Matrix_##arg
#define __NPE_MAP_TYPE(arg, ...) npe_Map_##arg

#define npe_function(name) void name() {
#define npe_arg(name, ...) \
  typedef npe::detail::__NPE_MATRIX_TYPE(__VA_ARGS__) __NPE_MATRIX_TYPE(name); \
  typedef npe::detail::__NPE_GET_FIRST(__VA_ARGS__) __NPE_MAP_TYPE(name);\
  npe::detail::__NPE_GET_FIRST(__VA_ARGS__) name;
#define npe_default_arg(name, ...) \
  typedef npe::detail::__NPE_MATRIX_TYPE(__VA_ARGS__) __NPE_MATRIX_TYPE(name); \
  typedef npe::detail::__NPE_GET_FIRST(__VA_ARGS__) __NPE_MAP_TYPE(name);\
  npe::detail::__NPE_GET_FIRST(__VA_ARGS__) name;
#define npe_begin_code()
#define npe_end_code() }
#define npe_matches(x) dense_float
#define npe_doc(docstr)
#define npe_sparse_like(x) sparse_float
#define npe_dense_like(x) dense_float
#endif // ifndef __NPE_FOR_REAL__

#endif // NPE_H

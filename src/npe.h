#ifndef NPE_H
#define NPE_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <npe_utils.h>
#include <npe_typedefs.h>
#include <npe_sparse_array.h>

#include <Eigen/Sparse>
#include <Eigen/Core>

// We define a bunch of stuff to make your ide work with npe_* syntax and produce valid types.
// When we actually generate the bindings all of this is gone
#ifndef __NPE_FOR_REAL__

namespace npe {
  namespace detail {

    typedef Eigen::Matrix<std::double_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_f64;
    typedef Eigen::Matrix<std::float_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_f32;
    typedef Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_i8;
    typedef Eigen::Matrix<std::int16_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_i16;
    typedef Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_i32;
    typedef Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_i64;
    typedef Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_u8;
    typedef Eigen::Matrix<std::uint16_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_u16;
    typedef Eigen::Matrix<std::uint32_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_u32;
    typedef Eigen::Matrix<std::uint64_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_dense_u64;

    typedef Eigen::Matrix<std::double_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_f64;
    typedef Eigen::Matrix<std::float_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_f32;
    typedef Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_i8;
    typedef Eigen::Matrix<std::int16_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_i16;
    typedef Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_i32;
    typedef Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_i64;
    typedef Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_u8;
    typedef Eigen::Matrix<std::uint16_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_u16;
    typedef Eigen::Matrix<std::uint32_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_u32;
    typedef Eigen::Matrix<std::uint64_t, Eigen::Dynamic, Eigen::Dynamic> npe_Matrix_sparse_u64;


    typedef Eigen::Map<npe_Matrix_dense_f64, Eigen::Aligned> sparse_f64;
    typedef Eigen::Map<npe_Matrix_dense_f32, Eigen::Aligned> sparse_f32;
    typedef Eigen::Map<npe_Matrix_dense_i8, Eigen::Aligned> sparse_i8;
    typedef Eigen::Map<npe_Matrix_dense_i16, Eigen::Aligned> sparse_i16;
    typedef Eigen::Map<npe_Matrix_dense_i32, Eigen::Aligned> sparse_i32;
    typedef Eigen::Map<npe_Matrix_dense_i64, Eigen::Aligned> sparse_i64;
    typedef Eigen::Map<npe_Matrix_dense_u8, Eigen::Aligned> sparse_u8;
    typedef Eigen::Map<npe_Matrix_dense_u16, Eigen::Aligned> sparse_u16;
    typedef Eigen::Map<npe_Matrix_dense_u32, Eigen::Aligned> sparse_u32;
    typedef Eigen::Map<npe_Matrix_dense_u64, Eigen::Aligned> sparse_u64;


    typedef Eigen::Map<npe_Matrix_sparse_f64, Eigen::Aligned> dense_f64;
    typedef Eigen::Map<npe_Matrix_sparse_f32, Eigen::Aligned> dense_f32;
    typedef Eigen::Map<npe_Matrix_sparse_i8, Eigen::Aligned> dense_i8;
    typedef Eigen::Map<npe_Matrix_sparse_i16, Eigen::Aligned> dense_i16;
    typedef Eigen::Map<npe_Matrix_sparse_i32, Eigen::Aligned> dense_i32;
    typedef Eigen::Map<npe_Matrix_sparse_i64, Eigen::Aligned> dense_i64;
    typedef Eigen::Map<npe_Matrix_sparse_u8, Eigen::Aligned> dense_u8;
    typedef Eigen::Map<npe_Matrix_sparse_u16, Eigen::Aligned> dense_u16;
    typedef Eigen::Map<npe_Matrix_sparse_u32, Eigen::Aligned> dense_u32;
    typedef Eigen::Map<npe_Matrix_sparse_u64, Eigen::Aligned> dense_u64;
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
#define npe_matches(x) dense_f32

#endif // __NPE_FOR_REAL__

#endif // NPE_H

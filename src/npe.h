#ifndef NPE_H
#define NPE_H

#include <npe_utils.h>

// We define a bunch of stuff to make your ide work with npe_* syntax and produce valid types.
// When we actually generate the bindings all of this is gone
#ifndef NPE_FOR_REAL
#define __NPE_GET_FIRST(arg, ...) arg
#define __NPE_MATRIX_TYPE(arg, ...) npe_Matrix_##arg
#define __NPE_MAP_TYPE(arg, ...) npe_Map_##arg

namespace npe {
namespace detail {

typedef Eigen::Matrix<std::double_t, Eigen::Dynamic, Eigen::Dynamic> matrix_dense_f64;
typedef Eigen::Map<matrix_dense_f64, Eigen::Aligned> dense_f64;

}
}

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
#endif // NPE_NOT_FOR_REAL

#endif // NPE_H

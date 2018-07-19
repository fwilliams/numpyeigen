#include <Eigen/Core>
#include <complex>

#ifndef EIGEN_TYPEDEFS_H
#define EIGEN_TYPEDEFS_H

enum NumpyTypeChar {
  char_float16  = 'e',
  char_float32  = 'f',
  char_float64  = 'd',
  char_float128 = 'g',

  char_int8  = 'b',
  char_int16 = 'h',
  char_int32 = 'i',
  char_int64 = 'l',

  char_uint8  = 'B',
  char_uint16 = 'H',
  char_uint32 = 'I',
  char_uint64 = 'L',

  char_complex64  = 'F',
  char_complex128 = 'D',
  char_complex256 = 'G',

  char_object = 'O',
  char_bytes_ = 'S',
  char_unicode = 'U',
  char_void_ = 'V',
};

enum NumpyTypeNum {
  num_float16  = 23,
  num_float32  = 11,
  num_float64  = 12,
  num_float128 = 13,

  num_int8  = 1,
  num_int16 = 3,
  num_int32 = 5,
  num_int64 = 7,

  num_uint8  = 2,
  num_uint16 = 4,
  num_uint32 = 6,
  num_uint64 = 8,

  num_complex64  = 14,
  num_complex128 = 15,
  num_complex256 = 16,

  num_object = 17,
  num_bytes_ = 18,
  num_unicode = 19,
  num_void_ = 20,
};

enum TypeId {
  // Row major floats
  type_f32_rm  = 0,
  type_f64_rm  = 1,
  type_f128_rm = 2,
  // Column major floats
  type_f32_cm  = 3,
  type_f64_cm  = 4,
  type_f128_cm = 5,
  // Non contiguous floats
  type_f32_x   = 6,
  type_f64_x   = 7,
  type_f128_x  = 8,


  // Row major signed ints
  type_i8_rm   = 9,
  type_i16_rm  = 10,
  type_i32_rm  = 11,
  type_i64_rm  = 12,
  // Column major signed ints
  type_i8_cm   = 13,
  type_i16_cm  = 14,
  type_i32_cm  = 15,
  type_i64_cm  = 16,
  // Non contiguous signed ints
  type_i8_x    = 17,
  type_i16_x   = 18,
  type_i32_x   = 19,
  type_i64_x   = 20,


  // Row Major unsigned ints
  type_u8_rm   = 21,
  type_u16_rm  = 22,
  type_u32_rm  = 23,
  type_u64_rm  = 24,
  // Column major unsigned ints
  type_u8_cm   = 25,
  type_u16_cm  = 26,
  type_u32_cm  = 27,
  type_u64_cm  = 28,
  // Non contiguous unsigned ints
  type_u8_x    = 29,
  type_u16_x   = 30,
  type_u32_x   = 31,
  type_u64_x   = 32,


  // Row Major complex floats
  type_c64_rm  = 33,
  type_c128_rm = 34,
  type_c256_rm = 35,
  // Column major complex floats
  type_c64_cm  = 36,
  type_c128_cm = 37,
  type_c256_cm = 38,
  // Non contiguous complex floats
  type_c64_x   = 39,
  type_c128_x  = 40,
  type_c256_x  = 41,
};

enum StorateOrder {
  ColMajor = 0,
  RowMajor = 1,
  NoContig = 2,
};

int get_type_id(char typechar, StorateOrder so) {
  using namespace  std;
  switch(so) {
  case ColMajor:
    switch (typechar) {
    case char_float32:
      return type_f32_cm;
    case char_float64:
      return type_f64_cm;
    case char_int8:
      return type_i8_cm;
    case char_int16:
      return type_i16_cm;
    case char_int32:
      return type_i32_cm;
    case char_int64:
      return type_i64_cm;
    case char_uint8:
      return type_u8_cm;
    case char_uint16:
      return type_u16_cm;
    case char_uint32:
      return type_u32_cm;
    case char_uint64:
      return type_u64_cm;
    case char_complex64:
      return type_c64_cm;
    case char_complex128:
      return type_c128_cm;
    case char_complex256:
      return type_c256_cm;
    default:
      cerr << "Bad Typechar" << endl;
      return -1;
    }
  case RowMajor:
    switch (typechar) {
    case char_float32:
      return type_f32_rm;
    case char_float64:
      return type_f64_rm;
    case char_int8:
      return type_i8_rm;
    case char_int16:
      return type_i16_rm;
    case char_int32:
      return type_i32_rm;
    case char_int64:
      return type_i64_rm;
    case char_uint8:
      return type_u8_rm;
    case char_uint16:
      return type_u16_rm;
    case char_uint32:
      return type_u32_rm;
    case char_uint64:
      return type_u64_rm;
    case char_complex64:
      return type_c64_rm;
    case char_complex128:
      return type_c128_rm;
    case char_complex256:
      return type_c256_rm;
    default:
      cerr << "Bad Typechar" << endl;
      return -1;
    }
  case NoContig:
    switch (typechar) {
    case char_float32:
      return type_f32_x;
    case char_float64:
      return type_f64_x;
    case char_int8:
      return type_i8_x;
    case char_int16:
      return type_i16_x;
    case char_int32:
      return type_i32_x;
    case char_int64:
      return type_i64_x;
    case char_uint8:
      return type_u8_x;
    case char_uint16:
      return type_u16_x;
    case char_uint32:
      return type_u32_x;
    case char_uint64:
      return type_u64_x;
    case char_complex64:
      return type_c64_x;
    case char_complex128:
      return type_c128_x;
    case char_complex256:
      return type_c256_x;
    default:
      cerr << "Bad Typechar" << endl;
      return -1;
    }
  default:
    cerr << "Bad StorageOrder" << endl;
    return -1;
  }
}

// IEEE Float matrix types
// There is no native float16 type so we probably won't support half floats
typedef Eigen::Matrix<
    std::float_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_f32_rm;
typedef Eigen::Matrix<
    std::float_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_f32_cm;
typedef Eigen::Matrix<
    std::float_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_f32_x;
typedef Eigen::Matrix<
    std::double_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_f64_rm;
typedef Eigen::Matrix<
    std::double_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_f64_cm;
typedef Eigen::Matrix<
    std::double_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_f64_x;
// TODO: Check if sizeof(long double) == 16 and only include this if true
//typedef Eigen::Matrix<
//    long double,
//    Eigen::Dynamic,
//    Eigen::Dynamic,
//    Eigen::RowMajor> Type_f128_rm;
//typedef Eigen::Matrix<
//    long double,
//    Eigen::Dynamic,
//    Eigen::Dynamic,
//    Eigen::ColMajor> Type_f128_cm;


// Signed integer matrix types
typedef Eigen::Matrix<
    std::int8_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_i8_rm;
typedef Eigen::Matrix<
    std::int8_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_i8_cm;
typedef Eigen::Matrix<
    std::int8_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_i8_x;
typedef Eigen::Matrix<
    std::int16_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_i16_rm;
typedef Eigen::Matrix<
    std::int16_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_i16_cm;
typedef Eigen::Matrix<
    std::int16_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_i16_x;
typedef Eigen::Matrix<
    std::int32_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_i32_rm;
typedef Eigen::Matrix<
    std::int32_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_i32_cm;
typedef Eigen::Matrix<
    std::int32_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_i32_x;
typedef Eigen::Matrix<
    std::int64_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_i64_rm;
typedef Eigen::Matrix<
    std::int64_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_i64_cm;
typedef Eigen::Matrix<
    std::int64_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_i64_x;

// Unsigned integer matrix types
typedef Eigen::Matrix<
    std::uint8_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_u8_rm;
typedef Eigen::Matrix<
    std::uint8_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_u8_cm;
typedef Eigen::Matrix<
    std::uint8_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_u8_x;
typedef Eigen::Matrix<
    std::uint16_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_u16_rm;
typedef Eigen::Matrix<
    std::uint16_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_u16_cm;
typedef Eigen::Matrix<
    std::uint16_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_u16_x;
typedef Eigen::Matrix<
    std::uint32_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_u32_rm;
typedef Eigen::Matrix<
    std::uint32_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_u32_cm;
typedef Eigen::Matrix<
    std::uint32_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_u32_x;
typedef Eigen::Matrix<
    std::uint64_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_u64_rm;
typedef Eigen::Matrix<
    std::uint64_t,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_u64_cm;
typedef Eigen::Matrix<
    std::uint64_t,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_u64_x;

// Complex float matrix types
typedef Eigen::Matrix<
    std::complex<float_t>,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_c64_rm;
typedef Eigen::Matrix<
    std::complex<float_t>,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_c64_cm;
typedef Eigen::Matrix<
    std::complex<float_t>,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_c64_x;
typedef Eigen::Matrix<
    std::complex<double_t>,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor> Type_c128_rm;
typedef Eigen::Matrix<
    std::complex<double_t>,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor> Type_c128_cm;
typedef Eigen::Matrix<
    std::complex<double_t>,
    Eigen::Dynamic,
    Eigen::Dynamic> Type_c128_x;
// TODO: Check if sizeof(long double) == 16 and only include this if true
//typedef Eigen::Matrix<
//    std::complex<long double>,
//    Eigen::Dynamic,
//    Eigen::Dynamic,
//    Eigen::RowMajor> Type_c256_rm;
//typedef Eigen::Matrix<
//    std::complex<long double>,
//    Eigen::Dynamic,
//    Eigen::Dynamic,
//    Eigen::ColMajor> Type_c256_cm;
#endif // EIGEN_TYPEDEFS_H

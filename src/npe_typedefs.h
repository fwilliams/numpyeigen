#ifndef BINDING_TYPEDEFS_H
#define BINDING_TYPEDEFS_H

#include <Eigen/Core>

#include <npe_sparse_array.h>

namespace npe {
namespace detail {

template <typename T>
struct is_sparse {
  static const bool value = false;
};

template <>
struct is_sparse<npe::sparse_array> {
  static const bool value = true;
};


enum NumpyTypeChar {
  char_f16  = 'e',
  char_f32  = 'f',
  char_f64  = 'd',
  char_f128 = 'g',

  char_i8  = 'b',
  char_i16 = 'h',
  char_i32 = 'i',
  char_i64 = 'l',
  char_i128 = 'q',

  char_u8  = 'B',
  char_u16 = 'H',
  char_u32 = 'I',
  char_u64 = 'L',
  char_u128 = 'Q',

  char_c64  = 'F',
  char_c128 = 'D',
  char_c256 = 'G',

  char_object = 'O',
  char_bytes_ = 'S',
  char_unicode = 'U',
  char_void_ = 'V',
};

enum NumpyTypeNum {
  num_f16  = 23,
  num_f32  = 11,
  num_f64  = 12,
  num_f128 = 13,

  num_i8  = 1,
  num_i16 = 3,
  num_i32 = 5,
  num_i64 = 7,
  num_i128 = 9,

  num_u8  = 2,
  num_u16 = 4,
  num_u32 = 6,
  num_u64 = 8,
  num_u128 = 10,

  num_c64  = 14,
  num_c128 = 15,
  num_c256 = 16,

  num_object = 17,
  num_bytes_ = 18,
  num_unicode = 19,
  num_void_ = 20,
};

enum TypeId {
  /*
   * Dense Types
   */
  // Row major floats
  dense_f32_rm  = 0,
  dense_f64_rm  = 1,
  dense_f128_rm = 2,
  // Column major floats
  dense_f32_cm  = 3,
  dense_f64_cm  = 4,
  dense_f128_cm = 5,
  // Non contiguous floats
  dense_f32_x   = 6,
  dense_f64_x   = 7,
  dense_f128_x  = 8,


  // Row major signed ints
  dense_i8_rm   = 9,
  dense_i16_rm  = 10,
  dense_i32_rm  = 11,
  dense_i64_rm  = 12,
  dense_i128_rm = 13,
  // Column major signed ints
  dense_i8_cm   = 14,
  dense_i16_cm  = 15,
  dense_i32_cm  = 16,
  dense_i64_cm  = 17,
  dense_i128_cm = 18,
  // Non contiguous signed ints
  dense_i8_x    = 19,
  dense_i16_x   = 20,
  dense_i32_x   = 21,
  dense_i64_x   = 22,
  dense_i128_x  = 23,


  // Row Major unsigned ints
  dense_u8_rm   = 24,
  dense_u16_rm  = 25,
  dense_u32_rm  = 26,
  dense_u64_rm  = 27,
  dense_u128_rm = 28,
  // Column major unsigned ints
  dense_u8_cm   = 29,
  dense_u16_cm  = 30,
  dense_u32_cm  = 31,
  dense_u64_cm  = 32,
  dense_u128_cm = 33,
  // Non contiguous unsigned ints
  dense_u8_x    = 34,
  dense_u16_x   = 35,
  dense_u32_x   = 36,
  dense_u64_x   = 37,
  dense_u128_x  = 38,

  // Row Major complex floats
  dense_c64_rm  = 39,
  dense_c128_rm = 40,
  dense_c256_rm = 41,
  // Column major complex floats
  dense_c64_cm  = 42,
  dense_c128_cm = 43,
  dense_c256_cm = 44,
  // Non contiguous complex floats
  dense_c64_x   = 45,
  dense_c128_x  = 46,
  dense_c256_x  = 47,


  /*
   * Sparse Types
   */
  // Row major floats
  sparse_f32_rm  = 48,
  sparse_f64_rm  = 49,
  sparse_f128_rm = 50,
  // Column major floats
  sparse_f32_cm  = 51,
  sparse_f64_cm  = 52,
  sparse_f128_cm = 53,
  // Non contiguous floats
  sparse_f32_x   = 54,
  sparse_f64_x   = 55,
  sparse_f128_x  = 56,


  // Row major signed ints
  sparse_i8_rm   = 57,
  sparse_i16_rm  = 58,
  sparse_i32_rm  = 59,
  sparse_i64_rm  = 60,
  sparse_i128_rm = 61,
  // Column major signed ints
  sparse_i8_cm   = 62,
  sparse_i16_cm  = 63,
  sparse_i32_cm  = 64,
  sparse_i64_cm  = 65,
  sparse_i128_cm = 66,
  // Non contiguous signed ints
  sparse_i8_x    = 67,
  sparse_i16_x   = 68,
  sparse_i32_x   = 69,
  sparse_i64_x   = 70,
  sparse_i128_x  = 71,


  // Row Major unsigned ints
  sparse_u8_rm   = 72,
  sparse_u16_rm  = 73,
  sparse_u32_rm  = 74,
  sparse_u64_rm  = 75,
  sparse_u128_rm = 76,
  // Column major unsigned ints
  sparse_u8_cm   = 77,
  sparse_u16_cm  = 78,
  sparse_u32_cm  = 79,
  sparse_u64_cm  = 80,
  sparse_u128_cm = 81,
  // Non contiguous unsigned ints
  sparse_u8_x    = 82,
  sparse_u16_x   = 83,
  sparse_u32_x   = 84,
  sparse_u64_x   = 85,
  sparse_u128_x  = 86,


  // Row Major complex floats
  sparse_c64_rm  = 87,
  sparse_c128_rm = 88,
  sparse_c256_rm = 89,
  // Column major complex floats
  sparse_c64_cm  = 90,
  sparse_c128_cm = 91,
  sparse_c256_cm = 92,
  // Non contiguous complex floats
  sparse_c64_x   = 93,
  sparse_c128_x  = 94,
  sparse_c256_x  = 95,
};

enum StorageOrder {
  ColMajor = Eigen::ColMajor,
  RowMajor = Eigen::RowMajor,
  NoOrder = Eigen::DontAlign,
};

enum Alignment {
  Aligned = Eigen::Aligned,
  Unaligned = Eigen::Unaligned
};

const std::string type_to_str(char type_char);
int get_type_id(bool is_sparse, char typechar, StorageOrder so);

} // namespace pybind
} // namespace igl

#endif // BINDING_TYPEDEFS_H

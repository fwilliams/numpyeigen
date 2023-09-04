#ifndef BINDING_TYPEDEFS_H
#define BINDING_TYPEDEFS_H

#include <Eigen/Core>

#include <npe_sparse_array.h>

// Check that we're compiling on a 64 bit platform
#if _WIN32 || _WIN64
#if _WIN64
static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
#else
static_assert(sizeof(int) == sizeof(long), "NumpyEigen Does not support 32-bit platforms");
static_assert(sizeof(unsigned int) == sizeof(unsigned long), "NumpyEigen Does not support 32-bit platforms");
#endif
#else
static_assert(sizeof(long) == sizeof(long long), "NumpyEigen Does not support 32-bit platforms");
static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "NumpyEigen Does not support 32-bit platforms");
#endif


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
  char_half  = 'e',
  char_float  = 'f',
  char_double  = 'd',
  char_longdouble = 'g',

  char_byte  = 'b',
  char_short = 'h',
  char_int = 'i',
  char_long = 'l',
  char_longlong = 'q',

  char_ubyte  = 'B',
  char_ushort = 'H',
  char_uint = 'I',
  char_ulong = 'L',
  char_ulonglong = 'Q',

  char_c64  = 'F',
  char_c128 = 'D',
  char_c256 = 'G',

  char_object = 'O',
  char_bytes_ = 'S',
  char_unicode = 'U',
  char_void_ = 'V',
  char_bool = '?',

  // not many letters left ¯\_(ツ)_/¯
  char_int32 = 'i',
  char_int64 = 'l',
  char_uint32 = 'I',
  char_uint64 = 'L',
};

enum NumpyTypeNum {
  num_half  = 23,
  num_float  = 11,
  num_double  = 12,
  num_longdouble = 13,

  num_byte  = 1,
  num_short = 3,
  num_int = 5,
  num_long = 7,
  num_longlong = 9,

  num_ubyte  = 2,
  num_ushort = 4,
  num_uint = 6,
  num_ulong = 8,
  num_ulonglong = 10,

  num_c64  = 14,
  num_c128 = 15,
  num_c256 = 16,

  num_object = 17,
  num_bytes_ = 18,
  num_unicode = 19,
  num_void_ = 20,
  num_bool = 21,

  // num_half defined to 23 above
  num_int32 = 24,
  num_int64 = 25,
  num_uint32 = 27,
  num_uint64 = 28,
};

enum TypeId {
  /*
   * Dense Types
   */
  // Row major floats
  dense_float_rm  = 0,
  dense_double_rm  = 1,
  dense_longdouble_rm = 2,
  // Column major floats
  dense_float_cm  = 3,
  dense_double_cm  = 4,
  dense_longdouble_cm = 5,
  // Non contiguous floats
  dense_float_x   = 6,
  dense_double_x   = 7,
  dense_longdouble_x  = 8,


  // Row major signed ints
  dense_byte_rm   = 9,
  dense_short_rm  = 10,
  // Column major signed ints
  dense_byte_cm   = 14,
  dense_short_cm  = 15,
  // Non contiguous signed ints
  dense_byte_x    = 19,
  dense_short_x   = 20,


  // Row Major unsigned ints
  dense_ubyte_rm   = 24,
  dense_ushort_rm  = 25,
  // Column major unsigned ints
  dense_ubyte_cm   = 29,
  dense_ushort_cm  = 30,
  // Non contiguous unsigned ints
  dense_ubyte_x    = 34,
  dense_ushort_x   = 35,

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
  //
  // Row major signed ints
  dense_int32_rm  = 99,
  dense_int64_rm  = 100,
  // Column major signed ints
  dense_int32_cm  = 102,
  dense_int64_cm  = 103,
  // Non contiguous signed ints
  dense_int32_x   = 105,
  dense_int64_x   = 106,

  // Row Major unsigned ints
  dense_uint32_rm  = 108,
  dense_uint64_rm  = 109,
  // Column major unsigned ints
  dense_uint32_cm  = 111,
  dense_uint64_cm  = 112,
  // Non contiguous unsigned ints
  dense_uint32_x   = 114,
  dense_uint64_x   = 115,


  /*
   * Sparse Types
   */
  // Row major floats
  sparse_float_rm  = 48,
  sparse_double_rm  = 49,
  sparse_longdouble_rm = 50,
  // Column major floats
  sparse_float_cm  = 51,
  sparse_double_cm  = 52,
  sparse_longdouble_cm = 53,
  // Non contiguous floats
  sparse_float_x   = 54,
  sparse_double_x   = 55,
  sparse_longdouble_x  = 56,


  // Row major signed ints
  sparse_byte_rm   = 57,
  sparse_short_rm  = 58,
  // Column major signed ints
  sparse_byte_cm   = 62,
  sparse_short_cm  = 63,
  // Non contiguous signed ints
  sparse_byte_x    = 67,
  sparse_short_x   = 68,


  // Row Major unsigned ints
  sparse_uint32_rm  = 117,
  sparse_uint64_rm  = 118,
  // Column major unsigned ints
  sparse_uint32_cm  = 120,
  sparse_uint64_cm  = 121,
  // Non contiguous unsigned ints
  sparse_uint32_x   = 123,
  sparse_uint64_x   = 124,

  // Row major signed ints
  sparse_int32_rm  = 126,
  sparse_int64_rm  = 127,
  // Column major signed ints
  sparse_int32_cm  = 129,
  sparse_int64_cm  = 130,
  // Non contiguous signed ints
  sparse_int32_x   = 132,
  sparse_int64_x   = 133,


  // Row Major unsigned ints
  sparse_ubyte_rm   = 72,
  sparse_ushort_rm  = 73,
  // Column major unsigned ints
  sparse_ubyte_cm   = 77,
  sparse_ushort_cm  = 78,
  // Non contiguous unsigned ints
  sparse_ubyte_x    = 82,
  sparse_ushort_x   = 83,

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

  // Row major bools
  dense_bool_rm = 96,
  // Column major bools
  dense_bool_cm = 97,
  // Non contiguous bools
  dense_bool_x = 98,

  // Alec: Suspicious that hese have the same values as dense_bool_*
  // Row major bools
  sparse_bool_rm = 96,
  // Column major bools
  sparse_bool_cm = 97,
  // Non contiguous bools
  sparse_bool_x = 98,
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
const std::string storage_order_to_str(StorageOrder so);
int get_type_id(bool is_sparse, char typechar, StorageOrder so);

} // namespace detail
} // namespace npe

#endif // BINDING_TYPEDEFS_H

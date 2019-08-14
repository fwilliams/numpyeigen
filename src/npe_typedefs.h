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
  dense_int_rm  = 11,
  dense_long_rm  = 12,
  dense_longlong_rm = 13,
  // Column major signed ints
  dense_byte_cm   = 14,
  dense_short_cm  = 15,
  dense_int_cm  = 16,
  dense_long_cm  = 17,
  dense_longlong_cm = 18,
  // Non contiguous signed ints
  dense_byte_x    = 19,
  dense_short_x   = 20,
  dense_int_x   = 21,
  dense_long_x   = 22,
  dense_longlong_x  = 23,


  // Row Major unsigned ints
  dense_ubyte_rm   = 24,
  dense_ushort_rm  = 25,
  dense_uint_rm  = 26,
  dense_ulong_rm  = 27,
  dense_ulonglong_rm = 28,
  // Column major unsigned ints
  dense_ubyte_cm   = 29,
  dense_ushort_cm  = 30,
  dense_uint_cm  = 31,
  dense_ulong_cm  = 32,
  dense_ulonglong_cm = 33,
  // Non contiguous unsigned ints
  dense_ubyte_x    = 34,
  dense_ushort_x   = 35,
  dense_uint_x   = 36,
  dense_ulong_x   = 37,
  dense_ulonglong_x  = 38,

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
  sparse_int_rm  = 59,
  sparse_long_rm  = 60,
  sparse_longlong_rm = 61,
  // Column major signed ints
  sparse_byte_cm   = 62,
  sparse_short_cm  = 63,
  sparse_int_cm  = 64,
  sparse_long_cm  = 65,
  sparse_longlong_cm = 66,
  // Non contiguous signed ints
  sparse_byte_x    = 67,
  sparse_short_x   = 68,
  sparse_int_x   = 69,
  sparse_long_x   = 70,
  sparse_longlong_x  = 71,


  // Row Major unsigned ints
  sparse_ubyte_rm   = 72,
  sparse_ushort_rm  = 73,
  sparse_uint_rm  = 74,
  sparse_ulong_rm  = 75,
  sparse_ulonglong_rm = 76,
  // Column major unsigned ints
  sparse_ubyte_cm   = 77,
  sparse_ushort_cm  = 78,
  sparse_uint_cm  = 79,
  sparse_ulong_cm  = 80,
  sparse_ulonglong_cm = 81,
  // Non contiguous unsigned ints
  sparse_ubyte_x    = 82,
  sparse_ushort_x   = 83,
  sparse_uint_x   = 84,
  sparse_ulong_x   = 85,
  sparse_ulonglong_x  = 86,


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

//FIXME: Use __cplusplus
#if __cplusplus < 201402L
const char transform_typechar(char t);
#else
constexpr char transform_typechar(char t) {

#ifdef _WIN64
    static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
    static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
    if (t == char_uint || t == char_ulong) {
        return char_uint;
    }
    if (t == char_int || t == char_long) {
        return char_int;
    }
#else
    static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
    static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned long long)");
    if (t == char_ulonglong || t == char_ulong) {
        return char_ulong;
    }
    if (t == char_long || t == char_longlong) {
        return char_long;
    }
#endif
    return t;
}
#endif

#if __cplusplus < 201402L
const int transform_typeid(int t);
#else
constexpr int transform_typeid(int t) {
#ifdef _WIN64
    static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
    static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
    switch(t){
    case dense_int_rm:
    case dense_long_rm:
        return dense_int_rm;
    case dense_int_cm:
    case dense_long_cm:
        return dense_int_cm;
    case dense_int_x:
    case dense_long_x:
        return dense_int_x;

    case dense_uint_rm:
    case dense_ulong_rm:
        return dense_uint_rm;
    case dense_uint_cm:
    case dense_ulong_cm:
        return dense_uint_cm;
    case dense_uint_x:
    case dense_ulong_x:
        return dense_uint_x;



    case sparse_int_rm:
    case sparse_long_rm:
        return sparse_int_rm;
    case sparse_int_cm:
    case sparse_long_cm:
        return sparse_int_cm;

    case sparse_uint_rm:
    case sparse_ulong_rm:
        return sparse_uint_rm;
    case sparse_uint_cm:
    case sparse_ulong_cm:
        return sparse_uint_cm;
    default:
        return t;
    }
#else
    static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
    static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned long long)");
    switch(t) {
    case dense_long_rm:
    case dense_longlong_rm:
        return dense_long_rm;
    case dense_long_cm:
    case dense_longlong_cm:
        return dense_long_cm;
    case dense_long_x:
    case dense_longlong_x:
        return dense_long_x;

    case dense_ulong_rm:
    case dense_ulonglong_rm:
        return dense_ulong_rm;
    case dense_ulong_cm:
    case dense_ulonglong_cm:
        return dense_ulong_cm;
    case dense_ulong_x:
    case dense_ulonglong_x:
        return dense_ulong_x;



    case sparse_long_rm:
    case sparse_longlong_rm:
        return sparse_long_rm;
    case sparse_long_cm:
    case sparse_longlong_cm:
        return sparse_long_cm;

    case sparse_ulong_rm:
    case sparse_ulonglong_rm:
        return sparse_ulong_rm;
    case sparse_ulong_cm:
    case sparse_ulonglong_cm:
        return sparse_ulong_cm;
    default:
        return t;
    }
#endif
}
#endif

} // namespace detail
} // namespace npe

#endif // BINDING_TYPEDEFS_H

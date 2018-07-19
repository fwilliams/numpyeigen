#include "numpyeigen_typedefs.h"

#include <iostream>


const std::string npe::detail::type_to_str(char type_char) {
  using namespace npe::detail;

  switch(type_char) {
  case char_f16:
    return "float16";
  case char_f32:
    return "float32";
  case char_f64:
    return "float64";
  case char_f128:
    return "float128";
  case char_i8:
    return "int8";
  case char_i16:
    return "int16";
  case char_i32:
    return "int32";
  case char_i64:
    return "int64";
  case char_u8:
    return "uint8";
  case char_u16:
    return "uint16";
  case char_u32:
    return "uint32";
  case char_u64:
    return "uint64";

  case char_c64:
    return "complex64";
  case char_c128:
    return "complex128";
  case char_c256:
    return "complex256";
  case char_object:
    return "object";
  case char_bytes_:
    return "bytes";
  case char_unicode:
    return "unicode";
  case char_void_:
    return "void";
  default:
    assert(false);
    return "";
  }
}


int npe::detail::get_type_id(char typechar, npe::detail::StorageOrder so) {
  using namespace  std;
  using namespace npe::detail;

  switch(so) {
  case ColMajor:
    switch (typechar) {
    case char_f32:
      return type_f32_cm;
    case char_f64:
      return type_f64_cm;
    case char_i8:
      return type_i8_cm;
    case char_i16:
      return type_i16_cm;
    case char_i32:
      return type_i32_cm;
    case char_i64:
      return type_i64_cm;
    case char_u8:
      return type_u8_cm;
    case char_u16:
      return type_u16_cm;
    case char_u32:
      return type_u32_cm;
    case char_u64:
      return type_u64_cm;
    case char_c64:
      return type_c64_cm;
    case char_c128:
      return type_c128_cm;
    case char_c256:
      return type_c256_cm;
    default:
      cerr << "Bad Typechar" << endl;
      return -1;
    }
  case RowMajor:
    switch (typechar) {
    case char_f32:
      return type_f32_rm;
    case char_f64:
      return type_f64_rm;
    case char_i8:
      return type_i8_rm;
    case char_i16:
      return type_i16_rm;
    case char_i32:
      return type_i32_rm;
    case char_i64:
      return type_i64_rm;
    case char_u8:
      return type_u8_rm;
    case char_u16:
      return type_u16_rm;
    case char_u32:
      return type_u32_rm;
    case char_u64:
      return type_u64_rm;
    case char_c64:
      return type_c64_rm;
    case char_c128:
      return type_c128_rm;
    case char_c256:
      return type_c256_rm;
    default:
      cerr << "Bad Typechar" << endl;
      return -1;
    }
  case NoOrder:
    switch (typechar) {
    case char_f32:
      return type_f32_x;
    case char_f64:
      return type_f64_x;
    case char_i8:
      return type_i8_x;
    case char_i16:
      return type_i16_x;
    case char_i32:
      return type_i32_x;
    case char_i64:
      return type_i64_x;
    case char_u8:
      return type_u8_x;
    case char_u16:
      return type_u16_x;
    case char_u32:
      return type_u32_x;
    case char_u64:
      return type_u64_x;
    case char_c64:
      return type_c64_x;
    case char_c128:
      return type_c128_x;
    case char_c256:
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

#include "npe_typedefs.h"

#include <iostream>

const std::string npe::detail::storage_order_to_str(npe::detail::StorageOrder so) {
    using namespace npe::detail;
    switch(so) {
    case StorageOrder::RowMajor:
        return std::string("Row Major");
    case StorageOrder::ColMajor:
        return std::string("Col Major");
    case StorageOrder::NoOrder:
        return std::string("No Major");
    default:
        return std::string("Corrupt Major");
    }
}

const std::string npe::detail::type_to_str(char type_char) {
  using namespace npe::detail;

  switch(type_char) {
  case char_half:
    return "half";
  case char_float:
    return "float";
  case char_double:
    return "double";
  case char_longdouble:
    return "longdouble";

  case char_byte:
    return "byte";
  case char_short:
    return "short";
  case char_ubyte:
    return "ubyte";
  case char_ushort:
    return "ushort";

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
  case char_int32:
    return "int32";
  case char_int64:
    return "int64";

  default:
    assert(false);
    return "";
  }
}


int npe::detail::get_type_id(bool is_sparse, char typechar, npe::detail::StorageOrder so) {
  using namespace  std;
  using namespace npe::detail;

  if (!is_sparse) {
    switch(so) {
    case ColMajor:
      switch (typechar) {
      case char_float:
        return dense_float_cm;
      case char_double:
        return dense_double_cm;
      case char_byte:
        return dense_byte_cm;
      case char_short:
        return dense_short_cm;
      case char_ubyte:
        return dense_ubyte_cm;
      case char_ushort:
        return dense_ushort_cm;
      case char_c64:
        return dense_c64_cm;
      case char_c128:
        return dense_c128_cm;
      case char_c256:
        return dense_c256_cm;
      case char_bool:
        return dense_bool_cm;
      case char_int32:
        return dense_int32_cm;
      case char_int64:
        return dense_int64_cm;
      case char_uint32:
        return dense_uint32_cm;
      case char_uint64:
        return dense_uint64_cm;
      default:
        cerr << "Bad Typechar '" << typechar << "'" << endl;
        return -1;
      }
    case RowMajor:
      switch (typechar) {
      case char_float:
        return dense_float_rm;
      case char_double:
        return dense_double_rm;
      case char_byte:
        return dense_byte_rm;
      case char_short:
        return dense_short_rm;
      case char_ubyte:
        return dense_ubyte_rm;
      case char_ushort:
        return dense_ushort_rm;
      case char_c64:
        return dense_c64_rm;
      case char_c128:
        return dense_c128_rm;
      case char_c256:
        return dense_c256_rm;
      case char_bool:
        return dense_bool_rm;
      case char_int32:
        return dense_int32_rm;
      case char_int64:
        return dense_int64_rm;
      case char_uint32:
        return dense_uint32_rm;
      case char_uint64:
        return dense_uint64_rm;
      default:
        cerr << "Bad Typechar '" << typechar << "'" << endl;
        return -1;
      }
    case NoOrder:
      switch (typechar) {
      case char_float:
        return dense_float_x;
      case char_double:
        return dense_double_x;
      case char_byte:
        return dense_byte_x;
      case char_short:
        return dense_short_x;
      case char_ubyte:
        return dense_ubyte_x;
      case char_ushort:
        return dense_ushort_x;
      case char_c64:
        return dense_c64_x;
      case char_c128:
        return dense_c128_x;
      case char_c256:
        return dense_c256_x;
      case char_bool:
        return dense_bool_x;
      case char_int32:
        return dense_int32_x;
      case char_int64:
        return dense_int64_x;
      case char_uint32:
        return dense_uint32_x;
      case char_uint64:
        return dense_uint64_x;
      default:
        cerr << "Bad Typechar '" << typechar << "'" << endl;
        return -1;
      }
    default:
      cerr << "Bad StorageOrder" << endl;
      return -1;
    }
  } else {
    switch(so) {
    case ColMajor:
      switch (typechar) {
      case char_float:
        return sparse_float_cm;
      case char_double:
        return sparse_double_cm;
      case char_byte:
        return sparse_byte_cm;
      case char_short:
        return sparse_short_cm;
      case char_ubyte:
        return sparse_ubyte_cm;
      case char_ushort:
        return sparse_ushort_cm;
      case char_c64:
        return sparse_c64_cm;
      case char_c128:
        return sparse_c128_cm;
      case char_c256:
        return sparse_c256_cm;
      case char_bool:
        return sparse_bool_cm;
      case char_int32:
        return sparse_int32_cm;
      case char_int64:
        return sparse_int64_cm;
      case char_uint32:
        return sparse_uint32_cm;
      case char_uint64:
        return sparse_uint64_cm;
      default:
        cerr << "Bad Typechar '" << typechar << "'" << endl;
        return -1;
      }
    case RowMajor:
      switch (typechar) {
      case char_float:
        return sparse_float_rm;
      case char_double:
        return sparse_double_rm;
      case char_byte:
        return sparse_byte_rm;
      case char_short:
        return sparse_short_rm;
      case char_ubyte:
        return sparse_ubyte_rm;
      case char_ushort:
        return sparse_ushort_rm;
      case char_c64:
        return sparse_c64_rm;
      case char_c128:
        return sparse_c128_rm;
      case char_c256:
        return sparse_c256_rm;
      case char_bool:
        return sparse_bool_rm;
      case char_int32:
        return sparse_int32_rm;
      case char_int64:
        return sparse_int64_rm;
      case char_uint32:
        return sparse_uint32_rm;
      case char_uint64:
        return sparse_uint64_rm;
      default:
        cerr << "Bad Typechar '" << typechar << "'" << endl;
        return -1;
      }
    default:
      cerr << "Bad StorageOrder" << endl;
      return -1;
    }
  }
}


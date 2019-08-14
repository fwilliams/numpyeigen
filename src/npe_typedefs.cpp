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
  case char_int:
    return "int";
  case char_long:
    return "long";
  case char_longlong:
    return "longlong";
  case char_ubyte:
    return "ubyte";
  case char_ushort:
    return "ushort";
  case char_uint:
    return "uint";
  case char_ulong:
    return "ulong";
  case char_ulonglong:
    return "ulonglong";

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
#if _WIN64
      static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
      case char_int:
      case char_long:
        return dense_int_cm;
      case char_longlong:
        return dense_longlong_cm;
#else
      static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
      case char_int:
        return dense_int_cm;
      case char_long:
      case char_longlong:
        return dense_long_cm;
#endif
      case char_ubyte:
        return dense_ubyte_cm;
      case char_ushort:
        return dense_ushort_cm;
#if _WIN64
      static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
      case char_uint:
      case char_ulong:
        return dense_uint_cm;
      case char_ulonglong:
        return dense_ulonglong_cm;
#else
      static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned long long)");
      case char_uint:
        return dense_uint_cm;
      case char_ulong:
      case char_ulonglong:
        return dense_ulong_cm;
#endif
      case char_c64:
        return dense_c64_cm;
      case char_c128:
        return dense_c128_cm;
      case char_c256:
        return dense_c256_cm;
      case char_bool:
        return dense_bool_cm;
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
#if _WIN64
      static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
      case char_int:
      case char_long:
        return dense_int_rm;
      case char_longlong:
        return dense_longlong_rm;
#else
      static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
      case char_int:
        return dense_int_rm;
      case char_long:
      case char_longlong:
        return dense_long_rm;
#endif
      case char_ubyte:
        return dense_ubyte_rm;
      case char_ushort:
        return dense_ushort_rm;
#if _WIN64
      static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
      case char_uint:
      case char_ulong:
        return dense_uint_rm;
      case char_ulonglong:
        return dense_ulonglong_rm;
#else
      static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned  long long)");
      case char_uint:
        return dense_uint_rm;
      case char_ulong:
      case char_ulonglong:
        return dense_ulong_rm;
#endif

      case char_c64:
        return dense_c64_rm;
      case char_c128:
        return dense_c128_rm;
      case char_c256:
        return dense_c256_rm;
      case char_bool:
        return dense_bool_rm;
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
#if _WIN64
      static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
      case char_int:
      case char_long:
        return dense_int_x;
      case char_longlong:
        return dense_longlong_x;
#else
      static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
      case char_int:
        return dense_int_x;
      case char_long:
      case char_longlong:
        return dense_long_x;
#endif
      case char_ubyte:
        return dense_ubyte_x;
      case char_ushort:
        return dense_ushort_x;
#if _WIN64
      static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
      case char_uint:
      case char_ulong:
        return dense_uint_x;
      case char_ulonglong:
        return dense_ulonglong_x;
#else
      static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned  long long)");
      case char_uint:
        return dense_uint_x;
      case char_ulong:
      case char_ulonglong:
        return dense_ulong_x;
#endif
      case char_c64:
        return dense_c64_x;
      case char_c128:
        return dense_c128_x;
      case char_c256:
        return dense_c256_x;
      case char_bool:
        return dense_bool_x;
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
#if _WIN64
      static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
      case char_int:
      case char_long:
        return sparse_int_cm;
      case char_longlong:
        return sparse_longlong_cm;
#else
      static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
      case char_int:
        return sparse_int_cm;
      case char_long:
      case char_longlong:
        return sparse_long_cm;
#endif
      case char_ubyte:
        return sparse_ubyte_cm;
      case char_ushort:
        return sparse_ushort_cm;
#if _WIN64
      static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
      case char_uint:
      case char_ulong:
        return sparse_uint_cm;
      case char_ulonglong:
        return sparse_ulonglong_cm;
#else
      static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned  long long)");
      case char_uint:
        return sparse_uint_cm;
      case char_ulong:
      case char_ulonglong:
        return sparse_ulong_cm;
#endif
      case char_c64:
        return sparse_c64_cm;
      case char_c128:
        return sparse_c128_cm;
      case char_c256:
        return sparse_c256_cm;
      case char_bool:
        return sparse_bool_cm;
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
#if _WIN64
      static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
      case char_int:
      case char_long:
        return sparse_int_rm;
      case char_longlong:
        return sparse_longlong_rm;
#else
      static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
      case char_int:
        return sparse_int_rm;
      case char_long:
      case char_longlong:
        return sparse_long_rm;
#endif
      case char_ubyte:
        return sparse_ubyte_rm;
      case char_ushort:
        return sparse_ushort_rm;
#if _WIN64
      static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
      case char_uint:
      case char_ulong:
        return sparse_uint_rm;
      case char_ulonglong:
        return sparse_ulonglong_rm;
#else
      static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned  long long)");
      case char_uint:
        return sparse_uint_rm;
      case char_ulong:
      case char_ulonglong:
        return sparse_ulong_rm;
#endif
      case char_c64:
        return sparse_c64_rm;
      case char_c128:
        return sparse_c128_rm;
      case char_c256:
        return sparse_c256_rm;
      case char_bool:
        return sparse_bool_rm;
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

#if __cplusplus < 201402L
const int npe::detail::transform_typeid(int t) {
#ifdef _WIN64
    static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
    static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
    switch(t){
    case npe::detail::dense_int_rm:
    case npe::detail::dense_long_rm:
        return npe::detail::dense_int_rm;
    case npe::detail::dense_int_cm:
    case npe::detail::dense_long_cm:
        return npe::detail::dense_int_cm;
    case npe::detail::dense_int_x:
    case npe::detail::dense_long_x:
        return npe::detail::dense_int_x;

    case npe::detail::dense_uint_rm:
    case npe::detail::dense_ulong_rm:
        return npe::detail::dense_uint_rm;
    case npe::detail::dense_uint_cm:
    case npe::detail::dense_ulong_cm:
        return npe::detail::dense_uint_cm;
    case npe::detail::dense_uint_x:
    case npe::detail::dense_ulong_x:
        return npe::detail::dense_uint_x;



    case npe::detail::sparse_int_rm:
    case npe::detail::sparse_long_rm:
        return npe::detail::sparse_int_rm;
    case npe::detail::sparse_int_cm:
    case npe::detail::sparse_long_cm:
        return npe::detail::sparse_int_cm;

    case npe::detail::sparse_uint_rm:
    case npe::detail::sparse_ulong_rm:
        return npe::detail::sparse_uint_rm;
    case npe::detail::sparse_uint_cm:
    case npe::detail::sparse_ulong_cm:
        return npe::detail::sparse_uint_cm;
    default:
        return t;
    }
#else
    static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
    static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned long long)");
    switch(t) {
    case npe::detail::dense_long_rm:
    case npe::detail::dense_longlong_rm:
        return npe::detail::dense_long_rm;
    case npe::detail::dense_long_cm:
    case npe::detail::dense_longlong_cm:
        return npe::detail::dense_long_cm;
    case npe::detail::dense_long_x:
    case npe::detail::dense_longlong_x:
        return npe::detail::dense_long_x;

    case npe::detail::dense_ulong_rm:
    case npe::detail::dense_ulonglong_rm:
        return npe::detail::dense_ulong_rm;
    case npe::detail::dense_ulong_cm:
    case npe::detail::dense_ulonglong_cm:
        return npe::detail::dense_ulong_cm;
    case npe::detail::dense_ulong_x:
    case npe::detail::dense_ulonglong_x:
        return npe::detail::dense_ulong_x;



    case npe::detail::sparse_long_rm:
    case npe::detail::sparse_longlong_rm:
        return npe::detail::sparse_long_rm;
    case npe::detail::sparse_long_cm:
    case npe::detail::sparse_longlong_cm:
        return npe::detail::sparse_long_cm;

    case npe::detail::sparse_ulong_rm:
    case npe::detail::sparse_ulonglong_rm:
        return npe::detail::sparse_ulong_rm;
    case npe::detail::sparse_ulong_cm:
    case npe::detail::sparse_ulonglong_cm:
        return npe::detail::sparse_ulong_cm;
    default:
        return t;
    }
#endif
}

const char npe::detail::transform_typechar(char t) {

#ifdef _WIN64
    static_assert(sizeof(int) == sizeof(long), "Expected sizeof(int) = sizeof(long) on 64 bit Windows");
    static_assert(sizeof(unsigned int) == sizeof(unsigned long), "Expected sizeof(unsigned int) = sizeof(unsigned long) on 64 bit Windows");
    if (t == npe::detail::char_uint || t == npe::detail::char_ulong) {
        return npe::detail::char_uint;
    }
    if (t == npe::detail::char_int || t == npe::detail::char_long) {
        return npe::detail::char_int;
    }
#else
    static_assert(sizeof(long) == sizeof(long long), "Expected sizeof(long) = sizeof(long long)");
    static_assert(sizeof(unsigned long) == sizeof(unsigned long long), "Expected sizeof(unsigned long) = sizeof(unsigned long long)");
    if (t == npe::detail::char_ulonglong || t == npe::detail::char_ulong) {
        return npe::detail::char_ulong;
    }
    if (t == npe::detail::char_long || t == npe::detail::char_longlong) {
        return npe::detail::char_long;
    }
#endif
    return t;
}
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifndef NPE_DTYPE_H
#define NPE_DTYPE_H

namespace npe {

enum DtypeType {
  type_f16     = 'e',
  type_f32     = 'f',
  type_f64     = 'd',
  type_f128    = 'g',

  type_i8      = 'b',
  type_i16     = 'h',
  type_i32     = 'i',
  type_i64     = 'l',
  type_i128    = 'q',

  type_u8      = 'B',
  type_u16     = 'H',
  type_u32     = 'I',
  type_u64     = 'L',
  type_u128    = 'Q',

  type_c64     = 'F',
  type_c128    = 'D',
  type_c256    = 'G',

  type_object  = 'O',
  type_bytes   = 'S',
  type_unicode = 'U',
  type_void    = 'V',
};

enum DtypeKind {
  kind_float     = 'f',
  kind_int       = 'i',
  kind_unsigned  = 'u',
  kind_complex   = 'c',
  kind_bool      = 'b',
  kind_ascii     = 'S',
  kind_unicode   = 'U',
  kind_arbitrary = 'V'
};


class dtype : public pybind11::object {
public:
    PYBIND11_OBJECT_DEFAULT(dtype, pybind11::object, pybind11::detail::npy_api::get().PyArrayDescr_Check_);

    explicit dtype(const pybind11::buffer_info &info) {
        dtype descr(_dtype_from_pep3118()(PYBIND11_STR_TYPE(info.format)));
        // If info.itemsize == 0, use the value calculated from the format string
        m_ptr = descr.strip_padding(info.itemsize ? info.itemsize : descr.itemsize()).release().ptr();
    }

    explicit dtype(const std::string &format) {
        m_ptr = from_args(pybind11::str(format)).release().ptr();
    }

    dtype(const char *format) : dtype(std::string(format)) { }

    dtype(pybind11::list names, pybind11::list formats, pybind11::list offsets, ssize_t itemsize) {
        pybind11::dict args;
        args["names"] = names;
        args["formats"] = formats;
        args["offsets"] = offsets;
        args["itemsize"] = pybind11::int_(itemsize);
        m_ptr = from_args(args).release().ptr();
    }

    /// This is essentially the same as calling numpy.dtype(args) in Python.
    static dtype from_args(pybind11::object args) {
        PyObject *ptr = nullptr;
        if (!pybind11::detail::npy_api::get().PyArray_DescrConverter_(args.release().ptr(), &ptr) || !ptr)
            throw pybind11::error_already_set();
        return pybind11::reinterpret_steal<dtype>(ptr);
    }

    /// Return dtype associated with a C++ type.
    template <typename T> static dtype of() {
        return pybind11::detail::npy_format_descriptor<typename std::remove_cv<T>::type>::dtype();
    }

    /// Size of the data type in bytes.
    ssize_t itemsize() const {
        return pybind11::detail::array_descriptor_proxy(m_ptr)->elsize;
    }

    /// Returns true for structured data types.
    bool has_fields() const {
        return pybind11::detail::array_descriptor_proxy(m_ptr)->names != nullptr;
    }

    /// Single-character type code.
    char kind() const {
        return pybind11::detail::array_descriptor_proxy(m_ptr)->kind;
    }

    /// Return the NumPy array type char
    char type() const {
        return pybind11::detail::array_descriptor_proxy(m_ptr)->type;
    }

    char byteorder() const {
        return pybind11::detail::array_descriptor_proxy(m_ptr)->byteorder;
    }

    /// Return the NumPy array descr flags
    int flags() const {
        return pybind11::detail::array_descriptor_proxy(m_ptr)->flags;
    }

    int type_num() const {
      return pybind11::detail::array_descriptor_proxy(m_ptr)->type_num;
    }

    int elsize() const {
      return pybind11::detail::array_descriptor_proxy(m_ptr)->elsize;
    }

    int alignment() const {
      return pybind11::detail::array_descriptor_proxy(m_ptr)->alignment;
    }
private:
    static pybind11::object _dtype_from_pep3118() {
        static PyObject *obj = pybind11::module::import("numpy.core._internal")
            .attr("_dtype_from_pep3118").cast<object>().release().ptr();
        return pybind11::reinterpret_borrow<object>(obj);
    }

    dtype strip_padding(ssize_t itemsize) {
        // Recursively strip all void fields with empty names that are generated for
        // padding fields (as of NumPy v1.11).
        if (!has_fields())
            return *this;

        struct field_descr { PYBIND11_STR_TYPE name; pybind11::object format; pybind11::int_ offset; };
        std::vector<field_descr> field_descriptors;

        for (auto field : attr("fields").attr("items")()) {
            auto spec = field.cast<pybind11::tuple>();
            auto name = spec[0].cast<pybind11::str>();
            auto format = spec[1].cast<pybind11::tuple>()[0].cast<dtype>();
            auto offset = spec[1].cast<pybind11::tuple>()[1].cast<pybind11::int_>();
            if (!len(name) && format.kind() == 'V')
                continue;
            field_descriptors.push_back({(PYBIND11_STR_TYPE) name, format.strip_padding(format.itemsize()), offset});
        }

        std::sort(field_descriptors.begin(), field_descriptors.end(),
                  [](const field_descr& a, const field_descr& b) {
                      return a.offset.cast<int>() < b.offset.cast<int>();
                  });

        pybind11::list names, formats, offsets;
        for (auto& descr : field_descriptors) {
            names.append(descr.name);
            formats.append(descr.format);
            offsets.append(descr.offset);
        }
        return dtype(names, formats, offsets, itemsize);
    }
};
}

namespace pybind11 {
namespace detail {

template <>
struct type_caster<npe::dtype> {
public:
  PYBIND11_TYPE_CASTER(npe::dtype, _("numpy.dtype | str | type"));

  bool load(handle src, bool) {
    if (!src) {
      return false;
    }

    object np = module::import("numpy");
    object np_dtype = np.attr("dtype");
    value = np_dtype(src);
    return true;
  }

  static handle cast(npe::dtype src, return_value_policy, handle) {
    return src.release();
  }
};

}
}
#endif // NPE_DTYPE_H

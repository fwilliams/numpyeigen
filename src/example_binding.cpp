#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>

#include <igl/cotmatrix.h>
#include <iostream>
#include <tuple>

#include "eigen_typedefs.h"

namespace py = pybind11;
using namespace std;

#define CRAZY1
#define CRAZY2
#define CRAZY3
#define CRAZY4
#define CRAZY5

#ifdef CRAZY1
#include "crazy_binding1.h"
#endif

#ifdef CRAZY2
#include "crazy_binding2.h"
#endif

#ifdef CRAZY3
#include "crazy_binding3.h"
#endif

#ifdef CRAZY4
#include "crazy_binding_4.h"
#endif

#ifdef CRAZY5
#include "crazy_binding_5.h"
#endif

PYBIND11_MODULE(pyigl_proto, m) {
    m.doc() = R"pbdoc(
        Playground for prototyping libIGL functions
        -----------------------

        .. currentmodule:: pyigl_proto

        .. autosummary::
           :toctree: _generate

           array_info
           cotmatrx
    )pbdoc";


    m.def("array_info", [](py::array v) {
      pybind11::dtype v_type = v.dtype();
      int v_flags = v.flags(); // PyArrayObject.flags

      cout << "PyArrayObject.descr.kind: " << v_type.kind() << endl;
      cout << "PyArrayObject.descr.type: " << v_type.type() << endl;
      cout << "PyArrayObject.descr.byteorder: " << v_type.byteorder() << endl;
      cout << "PyArrayObject.descr.flags: " << v_type.flags() << endl;
      cout << "PyArrayObject.descr.type_num: " << v_type.type_num() << endl;
      cout << "PyArrayObject.descr.elsize: " << v_type.elsize() << endl;
      cout << "PyArrayObject.descr.alignment: " << v_type.alignment() << endl;
      cout << endl;
      cout << "PyArrayObject.flags: " << v_flags << endl;
      cout << "PyArrayObject.flags & NPY_ARRAY_C_CONTIGUOUS: " << (v_flags & NPY_ARRAY_C_CONTIGUOUS) << endl;
      cout << "PyArrayObject.flags & NPY_ARRAY_F_CONTIGUOUS: " << (v_flags & NPY_ARRAY_F_CONTIGUOUS) << endl;
      cout << "PyArrayObject.flags & NPY_ARRAY_OWNDATA: " << (v_flags & NPY_ARRAY_OWNDATA) << endl;
      cout << "PyArrayObject.flags & NPY_ARRAY_ALIGNED: " << (v_flags & NPY_ARRAY_ALIGNED) << endl;
      cout << "PyArrayObject.flags & NPY_ARRAY_WRITEABLE: " << (v_flags & NPY_ARRAY_WRITEABLE) << endl;
      cout << "PyArrayObject.flags & NPY_ARRAY_UPDATEIFCOPY: " << (v_flags & NPY_ARRAY_UPDATEIFCOPY) << endl;
    });

    m.def("type_lookup", [](py::array v) {
      const char t = v.dtype().type();
      StorateOrder so = (v.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);

      return get_type_id(t, so);
    });

    m.def("return_an_int", [](py::array v) {
      return v.flags();
    });

#ifdef CRAZY1
    m.def("cotmatrix_branch", [](py::array v, py::array f) {
      return crazy_ass_branching_cot_matrix(v, f);
    });
#endif

#ifdef CRAZY2
    m.def("cotmatrix_branch2", [](py::array v, py::array f) {
      return crazier_ass_branching_cot_matrix(v, f);
    });
#endif

#ifdef CRAZY3
    m.def("cotmatrix_branch3", [](py::array v, py::array f) {
      return craziest_ass_branching_cot_matrix(v, f);
    });
#endif

#ifdef CRAZY4
    m.def("cotmatrix_branch4", [](py::array v1, py::array v2, py::array v3, py::array v4) {
      return krazi_ass_branching_cot_matrix(v1, v2, v3, v4);
    });
#endif

#ifdef CRAZY5
    m.def("cotmatrix_branch5", [](py::array v1, py::array v2, py::array v3, py::array v4, py::array v5) {
      return krazier_ass_branching_cot_matrix(v1, v2, v3, v4, v5);
    });
#endif

    m.def("cotmatrix", [](py::array v, py::array f) {
      // we care about the following attributes for each py::array
      // type
      // row-major/column-major/neither
      // alignment
      //
      // So a unique descriptor for one type is:
      // (type_num, [row|col|dyn], aligned_[none|8|16|32|64|128])
      // |----> In practice we will not support unaligned types because that makes life difficult
      //
      // We need a way of mapping this to an integer for each type then doing a lookup to find the right overload
      // So we need the following:
      // (1) - Generate a unique integer per descriptor type
      // (2) - Generate a static table of void* function pointers which correspond to each possible overload (this could be huge)
      // (3) - Perfect hashing to map list of ids to function call
      //
      // Other things to note:
      //   - We only support up to 2 dimensions in Eigen (there are no tensors yet) so we should probably check that things are 2d or less
      //   - It's possible to have weird but constant strides, if the stride is not the "expected" stride, then use Eigen::Dynamic for now
      //   - Let's not support unaligned data.

      py::dtype v_dtype = v.dtype();
      int v_shape0 = v.shape()[0];
      int v_shape1 = v.shape()[1];
      std::float_t* v_data = (std::float_t*)(v.data(0));

      Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);

      int f_shape0 = f.shape()[0];
      int f_shape1 = f.shape()[1];
      std::int32_t* f_data = (std::int32_t*)(f.data(0));
      Eigen::Map<Type_i32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);

      Eigen::SparseMatrix<std::float_t> L;

      return std::make_tuple<int, int>(V.rows(), F.rows());
//      igl::cotmatrix(V, F, L);
//      return L.rows();
//      cout << "Output cotmatrix dims: " << L.rows() << " x " << L.cols() << endl;

//      igl::cotmatrix<std::float_t, std::int8_t>();
//      igl::cotmatrix<std::float_t, std::int16_t>();
//      igl::cotmatrix<std::float_t, std::int64_t>();
//      igl::cotmatrix<std::float_t, std::uint8_t>();
//      igl::cotmatrix<std::float_t, std::uint16_t>();
//      igl::cotmatrix<std::float_t, std::uint32_t>();
//      igl::cotmatrix<std::float_t, std::uint64_t>();

//      igl::cotmatrix<std::double_t, std::int8_t>();
//      igl::cotmatrix<std::double_t, std::int16_t>();
//      igl::cotmatrix<std::double_t, std::int32_t>();
//      igl::cotmatrix<std::double_t, std::int64_t>();
//      igl::cotmatrix<std::double_t, std::uint8_t>();
//      igl::cotmatrix<std::double_t, std::uint16_t>();
//      igl::cotmatrix<std::double_t, std::uint32_t>();
//      igl::cotmatrix<std::double_t, std::uint64_t>();
    });
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

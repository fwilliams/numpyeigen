m.def("matrix_add", [](pybind11::array a, pybind11::array b) {
  const char _IGL_PY_BINDING_a_type_s = a.dtype().type();
  const igl::pybind::StorageOrder _IGL_PY_BINDING_a_so = (a.flags() & NPY_ARRAY_F_CONTIGUOUS) ? igl::pybind::RowMajor : (a.flags() & NPY_ARRAY_C_CONTIGUOUS ? igl::pybind::ColMajor : igl::pybind::NoOrder);
  const int _IGL_PY_BINDING_a_t_id = igl::pybind::get_type_id(_IGL_PY_BINDING_a_type_s, _IGL_PY_BINDING_a_so);
  const char _IGL_PY_BINDING_b_type_s = b.dtype().type();
  const igl::pybind::StorageOrder _IGL_PY_BINDING_b_so = (b.flags() & NPY_ARRAY_F_CONTIGUOUS) ? igl::pybind::RowMajor : (b.flags() & NPY_ARRAY_C_CONTIGUOUS ? igl::pybind::ColMajor : igl::pybind::NoOrder);
  const int _IGL_PY_BINDING_b_t_id = igl::pybind::get_type_id(_IGL_PY_BINDING_b_type_s, _IGL_PY_BINDING_b_so);
if (_IGL_PY_BINDING_a_type_s != igl::pybind::NumpyTypeChar::char_f64) {
  throw std::invalid_argument("Invalid type (float64) for argument 'a'. Expected one of ['float64'].");
}
if (_IGL_PY_BINDING_a_t_id != _IGL_PY_BINDING_b_t_id) {
  std::string err_msg = std::string("Invalid type (") + igl::pybind::type_to_str(_IGL_PY_BINDING_b_type_s) + std::string(") for argument 'b'. Expected it to match argument 'a' which is of type ") + igl::pybind::type_to_str(_IGL_PY_BINDING_a_type_s) + std::string(".");
  throw std::invalid_argument(err_msg);
}
if (_IGL_PY_BINDING_a_t_id == igl::pybind::TypeId::type_f64_cm) {
{
  struct IGL_PY_TYPE_a {
    typedef double Scalar;
    enum Layout { Order = igl::pybind::StorageOrder::ColMajor};
    enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::ColMajor> Eigen_Type;
    typedef Eigen::Map<IGL_PY_TYPE_a::Eigen_Type, IGL_PY_TYPE_a::Aligned> Map_Type;  };
  struct IGL_PY_TYPE_b {
    typedef double Scalar;
    enum Layout { Order = igl::pybind::StorageOrder::ColMajor};
    enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::ColMajor> Eigen_Type;
    typedef Eigen::Map<IGL_PY_TYPE_b::Eigen_Type, IGL_PY_TYPE_b::Aligned> Map_Type;  };

using namespace std;

// The binding framework will define structs for each of the numpy-overloaded input variables.
// These structs contain typedefs and enums specifying the type and layout of the array available
// at *compile time*. They are always named IGL_PY_TYPE_<varname> where 'varname' is the name of the
// numpy overloaded variable.
//
// An example struct for the variable 'a' is:
// struct IGL_PY_TYPE_a {
//   // The type of scalar stored in the array
//   typedef float Scalar;
//
//   // The layout of the array (either row major or column major)
//   enum Layout { Order = igl::pybind::StorageOrder::ColMajor};
//
//   // Whether the data is aligned in memory or not
//   enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
//
//   // The Eigen::Matrix type matching the array
//   typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::ColMajor> Eigen_Type;
//
//   // The Eigen::Map type used to make the array's data available to Eigen
//   typedef Eigen::Map<IGL_PY_TYPE_a::Eigen_Type, IGL_PY_TYPE_a::Aligned> Map_Type; // The Eigen::Map
// };

typedef IGL_PY_TYPE_a::Scalar Scalar_a;
typedef IGL_PY_TYPE_a::Eigen_Type Matrix_a;
typedef IGL_PY_TYPE_a::Map_Type Map_a;

typedef IGL_PY_TYPE_b::Scalar Scalar_b;
typedef IGL_PY_TYPE_b::Eigen_Type Matrix_b;
typedef IGL_PY_TYPE_b::Map_Type Map_b;

// Here we are creating Eigen::Maps for each numpy input which create a view on the memory of the input pybind11::array
// We can pass these maps to eigen functions without performing any extra copies.

// pybind11::array are a thin wrapper around PyArrayObject* numpy arrays
// They expose a data() method returning a pointer to the data,
// a shape() method, which returns a vector of the size of each dimension,
// as well as a bunch of other metadata about storage and alignment.

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
Map_b B((Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

// We compute some return values using the types specified in the input. So, for example,
// if you pass in matrices of doubles, you get the result as a matrix of doubles
Matrix_a ret1 = A + B;

// TODO: Check that this is doing a move and not a copy
// MOVE_TO_NP returns a pybind11::array which takes ownership of the Eigen::Matrix passed as a second argument.
// The first argument is the type of the Eigen::Matrix
return MOVE_TO_NP(Matrix_a, ret1);


}
} else if (_IGL_PY_BINDING_a_t_id == igl::pybind::TypeId::type_f64_rm) {
{
  struct IGL_PY_TYPE_a {
    typedef double Scalar;
    enum Layout { Order = igl::pybind::StorageOrder::RowMajor};
    enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::RowMajor> Eigen_Type;
    typedef Eigen::Map<IGL_PY_TYPE_a::Eigen_Type, IGL_PY_TYPE_a::Aligned> Map_Type;  };
  struct IGL_PY_TYPE_b {
    typedef double Scalar;
    enum Layout { Order = igl::pybind::StorageOrder::RowMajor};
    enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::RowMajor> Eigen_Type;
    typedef Eigen::Map<IGL_PY_TYPE_b::Eigen_Type, IGL_PY_TYPE_b::Aligned> Map_Type;  };

using namespace std;

// The binding framework will define structs for each of the numpy-overloaded input variables.
// These structs contain typedefs and enums specifying the type and layout of the array available
// at *compile time*. They are always named IGL_PY_TYPE_<varname> where 'varname' is the name of the
// numpy overloaded variable.
//
// An example struct for the variable 'a' is:
// struct IGL_PY_TYPE_a {
//   // The type of scalar stored in the array
//   typedef float Scalar;
//
//   // The layout of the array (either row major or column major)
//   enum Layout { Order = igl::pybind::StorageOrder::ColMajor};
//
//   // Whether the data is aligned in memory or not
//   enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
//
//   // The Eigen::Matrix type matching the array
//   typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::ColMajor> Eigen_Type;
//
//   // The Eigen::Map type used to make the array's data available to Eigen
//   typedef Eigen::Map<IGL_PY_TYPE_a::Eigen_Type, IGL_PY_TYPE_a::Aligned> Map_Type; // The Eigen::Map
// };

typedef IGL_PY_TYPE_a::Scalar Scalar_a;
typedef IGL_PY_TYPE_a::Eigen_Type Matrix_a;
typedef IGL_PY_TYPE_a::Map_Type Map_a;

typedef IGL_PY_TYPE_b::Scalar Scalar_b;
typedef IGL_PY_TYPE_b::Eigen_Type Matrix_b;
typedef IGL_PY_TYPE_b::Map_Type Map_b;

// Here we are creating Eigen::Maps for each numpy input which create a view on the memory of the input pybind11::array
// We can pass these maps to eigen functions without performing any extra copies.

// pybind11::array are a thin wrapper around PyArrayObject* numpy arrays
// They expose a data() method returning a pointer to the data,
// a shape() method, which returns a vector of the size of each dimension,
// as well as a bunch of other metadata about storage and alignment.

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
Map_b B((Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

// We compute some return values using the types specified in the input. So, for example,
// if you pass in matrices of doubles, you get the result as a matrix of doubles
Matrix_a ret1 = A + B;

// TODO: Check that this is doing a move and not a copy
// MOVE_TO_NP returns a pybind11::array which takes ownership of the Eigen::Matrix passed as a second argument.
// The first argument is the type of the Eigen::Matrix
return MOVE_TO_NP(Matrix_a, ret1);


}
} else if (_IGL_PY_BINDING_a_t_id == igl::pybind::TypeId::type_f64_x) {
{
  struct IGL_PY_TYPE_a {
    typedef double Scalar;
    enum Layout { Order = igl::pybind::StorageOrder::NoOrder};
    enum Aligned { Aligned = igl::pybind::Alignment::Unaligned};
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::NoOrder> Eigen_Type;
    typedef Eigen::Map<IGL_PY_TYPE_a::Eigen_Type, IGL_PY_TYPE_a::Aligned> Map_Type;  };
  struct IGL_PY_TYPE_b {
    typedef double Scalar;
    enum Layout { Order = igl::pybind::StorageOrder::NoOrder};
    enum Aligned { Aligned = igl::pybind::Alignment::Unaligned};
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::NoOrder> Eigen_Type;
    typedef Eigen::Map<IGL_PY_TYPE_b::Eigen_Type, IGL_PY_TYPE_b::Aligned> Map_Type;  };

using namespace std;

// The binding framework will define structs for each of the numpy-overloaded input variables.
// These structs contain typedefs and enums specifying the type and layout of the array available
// at *compile time*. They are always named IGL_PY_TYPE_<varname> where 'varname' is the name of the
// numpy overloaded variable.
//
// An example struct for the variable 'a' is:
// struct IGL_PY_TYPE_a {
//   // The type of scalar stored in the array
//   typedef float Scalar;
//
//   // The layout of the array (either row major or column major)
//   enum Layout { Order = igl::pybind::StorageOrder::ColMajor};
//
//   // Whether the data is aligned in memory or not
//   enum Aligned { Aligned = igl::pybind::Alignment::Aligned};
//
//   // The Eigen::Matrix type matching the array
//   typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, igl::pybind::StorageOrder::ColMajor> Eigen_Type;
//
//   // The Eigen::Map type used to make the array's data available to Eigen
//   typedef Eigen::Map<IGL_PY_TYPE_a::Eigen_Type, IGL_PY_TYPE_a::Aligned> Map_Type; // The Eigen::Map
// };

typedef IGL_PY_TYPE_a::Scalar Scalar_a;
typedef IGL_PY_TYPE_a::Eigen_Type Matrix_a;
typedef IGL_PY_TYPE_a::Map_Type Map_a;

typedef IGL_PY_TYPE_b::Scalar Scalar_b;
typedef IGL_PY_TYPE_b::Eigen_Type Matrix_b;
typedef IGL_PY_TYPE_b::Map_Type Map_b;

// Here we are creating Eigen::Maps for each numpy input which create a view on the memory of the input pybind11::array
// We can pass these maps to eigen functions without performing any extra copies.

// pybind11::array are a thin wrapper around PyArrayObject* numpy arrays
// They expose a data() method returning a pointer to the data,
// a shape() method, which returns a vector of the size of each dimension,
// as well as a bunch of other metadata about storage and alignment.

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
Map_b B((Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

// We compute some return values using the types specified in the input. So, for example,
// if you pass in matrices of doubles, you get the result as a matrix of doubles
Matrix_a ret1 = A + B;

// TODO: Check that this is doing a move and not a copy
// MOVE_TO_NP returns a pybind11::array which takes ownership of the Eigen::Matrix passed as a second argument.
// The first argument is the type of the Eigen::Matrix
return MOVE_TO_NP(Matrix_a, ret1);


}
} else {
  throw std::invalid_argument("This should never happen but clearly it did. File github issue plz.");
}

});

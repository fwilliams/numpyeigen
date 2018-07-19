#include <tuple>
#include <Eigen/Core>
#include "numpyeigen_utils.h"

// This defines a function named 'test_binding' which will be exposed to python
igl_binding("test_binding")

// This says test binding takes as input a variable, 'a', which can have type type_f32 (float) or type_f64 (double)
// Since there are more than one type specified, the variable 'a' is a numpy overload. Numpy overloaded variables
// have the type pybind11::array whose scalar type must be one of the specified types (f32 or f64 in this case).
igl_input("a", "type_f32", "type_f64")

// This is another input variable, 'b'. It's type is a matches() statement. In this case, the matches() says that the
// type of 'b' must match the type of 'a'. matches() statements can only match to numpy overloaded variables.
igl_input("b", "matches(a)")

// Here is another variable 'c' and two variables whose types have to match it.
igl_input("c", "type_i32", "type_i64")
igl_input("d", "matches(c)")
igl_input("e", "matches(d)")

// This is a non-numpy overloaded input parameter of type int.
igl_input("f", "int")

// After this directive is the actual source code of the bound function
igl_begin_code()

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

typedef IGL_PY_TYPE_c::Scalar Scalar_c;
typedef IGL_PY_TYPE_c::Eigen_Type Matrix_c;
typedef IGL_PY_TYPE_c::Map_Type Map_c;

typedef IGL_PY_TYPE_d::Scalar Scalar_d;
typedef IGL_PY_TYPE_d::Eigen_Type Matrix_d;
typedef IGL_PY_TYPE_d::Map_Type Map_d;

typedef IGL_PY_TYPE_d::Scalar Scalar_e;
typedef IGL_PY_TYPE_d::Eigen_Type Matrix_e;
typedef IGL_PY_TYPE_d::Map_Type Map_e;

// Here we are creating Eigen::Maps for each numpy input which create a view on the memory of the input pybind11::array
// We can pass these maps to eigen functions without performing any extra copies.

// pybind11::array are a thin wrapper around PyArrayObject* numpy arrays
// They expose a data() method returning a pointer to the data,
// a shape() method, which returns a vector of the size of each dimension,
// as well as a bunch of other metadata about storage and alignment.

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
Map_b B((Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

Map_c C((Scalar_c*)c.data(), c.shape()[0], c.shape()[1]);
Map_d D((Scalar_d*)d.data(), d.shape()[0], d.shape()[1]);
Map_e E((Scalar_e*)e.data(), e.shape()[0], e.shape()[1]);

// We compute some return values using the types specified in the input. So, for example,
// if you pass in matrices of doubles, you get the result as a matrix of doubles
Matrix_a ret1 = A + B;
Matrix_c ret2 = C + D + E * f;

// TODO: Check that this is doing a move and not a copy
// MOVE_TO_NP returns a pybind11::array which takes ownership of the Eigen::Matrix passed as a second argument.
// The first argument is the type of the Eigen::Matrix
return std::make_tuple(MOVE_TO_NP(Matrix_a, ret1), MOVE_TO_NP(Matrix_c, ret2));

igl_end_code()



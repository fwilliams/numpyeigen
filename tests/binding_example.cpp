#include <tuple>
#include <Eigen/Core>
#include "numpyeigen_utils.h"

// This defines a function named 'test_binding' which will be exposed to python
npe_function("test_binding")

// This says test binding takes as input a variable, 'a', which can have type type_f32 (float) or type_f64 (double)
// Since there are more than one type specified, the variable 'a' is a numpy overload. Numpy overloaded variables
// have the type pybind11::array whose scalar type must be one of the specified types (f32 or f64 in this case).
npe_arg("a", "type_f32", "type_f64")

// This is another input variable, 'b'. It's type is a matches() statement. In this case, the matches() says that the
// type of 'b' must match the type of 'a'. matches() statements can only match to numpy overloaded variables.
npe_arg("b", "matches(a)")

// Here is another variable 'c' and two variables whose types have to match it.
npe_arg("c", "type_i32", "type_i64")
npe_arg("d", "matches(c)")
npe_arg("e", "matches(d)")

// This is a non-numpy overloaded input parameter of type int.
npe_arg("f", "int")

// After this directive is the actual source code of the bound function
npe_begin_code()

// The binding framework will define a structs containing type information for each of the numpy-overloaded input
// variables. This struct contains typedefs and enums specifying the type and layout of the array
// arguments. Specifically, for each argument, x, the following typedefs are available:
//
// * npe::Scalar_x - The scalar type of x
// * npe::Matrix_x - The Eigen::Matrix<> template specialization corresponding to x
// * npe::Map_x    - An Eigen::Map<> type specialized for x
//

// pybind11::array are a thin wrapper around PyArrayObject* numpy arrays
// They expose a data() method returning a pointer to the data,
// a shape() method, which returns a vector of the size of each dimension,
// as well as a bunch of other metadata about storage and alignment.

npe::Map_a A((npe::Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
npe::Map_b B((npe::Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);

npe::Map_c C((npe::Scalar_c*)c.data(), c.shape()[0], c.shape()[1]);
npe::Map_d D((npe::Scalar_d*)d.data(), d.shape()[0], d.shape()[1]);
npe::Map_e E((npe::Scalar_e*)e.data(), e.shape()[0], e.shape()[1]);

// We compute some return values using the types specified in the input. So, for example,
// if you pass in matrices of doubles, you get the result as a matrix of doubles
npe::Matrix_a ret1 = A + B;
npe::Matrix_c ret2 = C + D + E * f;

// MOVE_TO_NP returns a pybind11::array which takes ownership of the Eigen::Matrix passed as a second argument.
// The first argument is the type of the Eigen::Matrix
// return std::make_tuple(NPE_MOVE(npe::Matrix_a, ret1), NPE_MOVE(npe::Matrix_c, ret2));
return std::make_tuple(NPE_MOVE(ret1), NPE_MOVE(ret2));
npe_end_code()



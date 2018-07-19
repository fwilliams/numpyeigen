# NumpyEigen - Fast zero-overhead bindings between NumPy and Eigen
-----------------------

**NumpyEigen** makes it easy to transparently convert [NumPy](http://www.numpy.org/) arrays to [Eigen](https://www.google.com/search?client=ubuntu&channel=fs&q=eigen&ie=utf-8&oe=utf-8) with zero copy overhead while taking advantage of Eigen's expression template system for maximum performance.

[Eigen](https://www.google.com/search?client=ubuntu&channel=fs&q=eigen&ie=utf-8&oe=utf-8) is a C++ numerical linear algebra. It uses expression templates to pick the fastest numerical algorithms for a given set of input types. Numpy is a numerical library exposing fast numerical routines in Python. 

Since type information in Python is only available at runtime, it is not easy to write general bindings which accept multiple NumPy types, have zero copy overhead, and can make use of the fastest numerical kernels in Eigen. NumpyEigen transparently generates bindings which do all of the above, exposing numpy type information at compile time to C++ code. 

NumpyEigen comes built-in with [CMake](https://cmake.org/) tools to integrate with existing build systems in a single line of code. A set of scripts to integrate with other build systems will be included in the future.

## Example
See the repository [TODO: Example repo]() for a fully running example project.

When compiled, the following code will generate a function `foo(a, b, c, d, e)` callable from Python. `foo` returns a tuple of values computed from its inputs.

```c++
#include <tuple>
#include <string>

npe_function("foo");                  // create a function foo exposed to python

// The arguments to foo are as follows:
npe_arg("a", "type_f64", "type_f32"); // a is a numpy array with dtype either float or double
npe_arg("b", "matches(a)");           // b is a numpy array whose type has to match a
npe_arg("c", "type_i32", "type_i64"); // c is a numpy array whose type is either int32 or int64
npe_arg("d", "std::string");          // d is a string
npe_arg("e", "int");                  // e is an int

// The C++ code for the function starts after this line
npe_begin_code();

typedef NPE_PY_TYPE_a::Scalar Scalar_a;
typedef NPE_PY_TYPE_a::Eigen_Type Matrix_a;
typedef NPE_PY_TYPE_a::Map_Type Map_a;

typedef NPE_PY_TYPE_b::Scalar Scalar_b;
typedef NPE_PY_TYPE_b::Eigen_Type Matrix_b;
typedef NPE_PY_TYPE_b::Map_Type Map_b;

typedef NPE_PY_TYPE_c::Scalar Scalar_c;
typedef NPE_PY_TYPE_c::Eigen_Type Matrix_c;
typedef NPE_PY_TYPE_c::Map_Type Map_c;

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
Map_b B((Scalar_b*)b.data(), b.shape()[0], b.shape()[1]);
Map_c C((Scalar_c*)c.data(), c.shape()[0], c.shape()[1]);

Matrix_a ret1 = A + B;
Matrix_a ret2 = A - C;
int ret3 = d + std::string("concatenated");
int ret4 = e + 2;

return std::make_tuple(ret1, ret2, ret3, ret4, ret5);

npe_end_code()
```

To create a Python Module named `mymodule` which exposes the function foo, stored in foo.cpp add the following to your `CMakeLists.txt`:

```cmake
include(numpyeigenTools)  # This module is located in the `cmake` directory
npe_add_module(mymodule, BINDING_SOURCES foo.cpp)
```

## Building and running tests

Don't forget to `git clone --recursive`!!!

Building the tests should be as easy as:
```
mkdir build
cd build
cmake ..
make
make test

There's an `tests/environment.yml` file which can be used to generate conda environment with all the right dependencies by running:
```
conda env create -f environment.yml
```

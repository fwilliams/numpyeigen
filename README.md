# NumpyEigen - Fast zero-overhead bindings between NumPy and Eigen ![Build Status](https://travis-ci.com/fwilliams/numpyeigen.svg?branch=master)

**NumpyEigen** makes it easy to transparently convert [NumPy](http://www.numpy.org/) 
dense arrays and [SciPy](https://docs.scipy.org/doc/scipy/reference/sparse.html) sparse 
matrices to [Eigen](https://www.google.com/search?client=ubuntu&channel=fs&q=eigen&ie=utf-8&oe=utf-8) 
with zero copy overhead while taking advantage of Eigen's expression template system for maximum performance.

[Eigen](https://www.google.com/search?client=ubuntu&channel=fs&q=eigen&ie=utf-8&oe=utf-8) is a C++ numerical 
linear algebra librry. It uses expression templates to pick the fastest numerical algorithms for a given set of input 
types. NumPy and SciPy are librarieis exposing fast numerical routines in Python. 

Since type information in Python is only available at runtime, it is not easy to write bindings which accept 
multiple NumPy or SciPy types, have zero copy overhead, and can make use of the fastest numerical kernels in Eigen. 
NumpyEigen transparently generates bindings which do all of the above, exposing numpy type information at compile 
time to C++ code. 

### Zero copy overhead 
Bindings written with NumpyEigen have zero copy overhead from Python to C++ and vice versa for any NumPy dense array or 
SciPy CSR or CSC sparse matrix.

### Binding Function Overloading
NumpyEigen allows type overloading of binding input arguments. The type of the argument is made available at compile 
time to the C++ code. This type information is in turn used to drive Eigen's expression template system to choose 
the fastest numerical algorithm for a given input type.

### Build System Support
NumpyEigen comes built-in with [CMake](https://cmake.org/) tools to integrate with existing build systems in a
 single line of code. A set of scripts to integrate with other build systems will be included in the future.

### Minimal Dependencies
NumpyEigen only requires the system to have a valid C++ compiler and a running Python interpreter with version > 3.0. 

NumpyEigen uses [pybind11](https://github.com/pybind/pybind11) under the hood which is included as a submodule. 
Don't forget to `git clone --recursive`!

## Example
See the repository [TODO: Example repo](https://github.com/fwilliams/numpyeigen/issues/9) for a fully 
running example project.

When compiled, the following code will generate a function `foo(a, b, c, d, e, f)` callable from Python. 
`foo` returns a tuple of values computed from its inputs.

```c++
#include <npe.h>
#include <tuple>
#include <string>

// Create a function named foo exposed to python
npe_function(foo)                     

// The arguments to foo are as follows:
// Each of these are transparently converted from numpy types to appropriate Eigen::Map types
// wiith zero copy overhead.
npe_arg(a, dense_f64, dense_f32)       // a is a numpy array with dtype either float or double
npe_arg(b, matches(a))                // b is a numpy array whose type has to match a
npe_arg(c, dense_i32, dense_i64)      // c is a numpy array whose type is either int32 or int64
npe_arg(d, std::string)               // d is a string
npe_arg(f, sparse_f32, sparse_f64)    // f is a sparse matrix whose data is either float32 or float64
npe_arg(e, int)                       // e is an int

// NumpyEigen supports doc strings which are expression evaluating to C strings or std::string types
npe_doc("A function which computes various values from input matrices")

// The C++ code for the function starts after this line
npe_begin_code()

// npe_Matrix_* are Eigen::Matrix<T> or Eigen::SparseMatrix<T> types corresponding to the inputs
npe_Matrix_a ret1 = a + b;
npe_Matrix_a ret2 = a - c;
int ret3 = d + std::string("concatenated");
int ret4 = e + 2;
npe_Matrix_f ret5 = f * 1.5;

// npe::move() wraps an Eigen type in a NumPy or SciPy type with zero copy overhead
return std::make_tuple(npe::move(ret1), npe::move(ret2), ret3, ret4, npe::move(ret5));

npe_end_code()
```

To create a Python Module named `mymodule` which exposes the function foo, 
stored in foo.cpp add one of the following to your `CMakeLists.txt`:

#### 1) If NumpyEigen is a subdirectory of your project
```cmake
add_subdirectory(/path/to/numpyeigen)

npe_add_module(mymodule, BINDING_SOURCES foo.cpp)
```

#### 2) If NumpyEigen is not a subdirectory of your project
```cmake
# Make numpyeigen available in the current project
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /path/to/numpyeigen/cmake)
include(numpyeigen)

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
```

There's an `tests/environment.yml` file which can be used to generate conda environment with a
ll the right dependencies by running:
```
conda env create -f environment.yml
```

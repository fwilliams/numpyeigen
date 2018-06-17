#include <tuple>
#include <iostream>

#include <Eigen/Core>
#include <pybind11/numpy.h>

#include "eigen_typedefs.h"

#ifndef CRAZY_BINDING5_H
#define CRAZY_BINDING5_H

namespace py = pybind11;

std::tuple<int, int, int, int, int, int>
krazier_ass_branching_cot_matrix(py::array v1, py::array v2, py::array v3, py::array v4, py::array v5) {
  using namespace std;

  const char t1 = v1.dtype().type();
  const char t2 = v2.dtype().type();
  const char t3 = v3.dtype().type();
  const char t4 = v4.dtype().type();
  const char t5 = v5.dtype().type();
//  const char t6 = v6.dtype().type();
  StorateOrder so1 = (v1.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v1.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  StorateOrder so2 = (v2.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v2.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  StorateOrder so3 = (v3.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v3.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  StorateOrder so4 = (v4.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v4.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  StorateOrder so5 = (v5.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v5.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
//  StorateOrder so6 = (v6.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v6.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);

  int tid0 = get_type_id(t1, so1);
  int tid1 = get_type_id(t2, so2);
  int tid2 = get_type_id(t3, so3);
  int tid3 = get_type_id(t4, so4);
  int tid4 = get_type_id(t5, so5);
//  int tid5 = get_type_id(t6, so6);

  long int v1_shape[2] = { v1.shape()[0], v1.shape()[1] };
  long int v2_shape[2] = { v2.shape()[0], v2.shape()[1] };
  long int v3_shape[2] = { v3.shape()[0], v3.shape()[1] };
  long int v4_shape[2] = { v4.shape()[0], v4.shape()[1] };
  long int v5_shape[2] = { v5.shape()[0], v5.shape()[1] };
//  long int v6_shape[2] = { v6.shape()[0], v6.shape()[1] };

  long int v1_strides[2] = { v1.strides()[0], v1.strides()[1] };
  long int v2_strides[2] = { v2.strides()[0], v2.strides()[1] };
  long int v3_strides[2] = { v3.strides()[0], v3.strides()[1] };
  long int v4_strides[2] = { v4.strides()[0], v4.strides()[1] };
  long int v5_strides[2] = { v5.strides()[0], v5.strides()[1] };
//  long int v6_strides[2] = { v6.strides()[0], v6.strides()[1] };

  #include "codegen5.inc"

  //  return L.rows();
}


#endif // CRAZY_BINDING5_H

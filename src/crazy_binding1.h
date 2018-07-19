#include <tuple>
#include <iostream>

#include <Eigen/Core>

#include "eigen_typedefs.h"

#ifndef CRAZY_BINDING1_H
#define CRAZY_BINDING1_H


std::tuple<int, int> crazy_ass_branching_cot_matrix(py::array v, py::array f) {
  using namespace std;

  const char t1 = v.dtype().type();
  const char t2 = f.dtype().type();
  StorateOrder so1 = (v.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  StorateOrder so2 = (f.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (f.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  int tid1 = get_type_id(t1, so1);
  int tid2 = get_type_id(t2, so2);

  int v_shape0 = v.shape()[0];
  int v_shape1 = v.shape()[1];
  int f_shape0 = f.shape()[0];
  int f_shape1 = f.shape()[1];

  Eigen::SparseMatrix<double> L;

  if (tid1 == type_f32_cm && tid2 == type_i32_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_cm && tid2 == type_i64_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_cm && tid2 == type_u32_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_cm && tid2 == type_u64_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }
  else if (tid1 == type_f32_rm && tid2 == type_i32_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_rm && tid2 == type_i64_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_rm && tid2 == type_u32_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_rm && tid2 == type_u64_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }
  else if (tid1 == type_f32_cm && tid2 == type_i32_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_cm && tid2 == type_i64_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_cm && tid2 == type_u32_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_cm && tid2 == type_u64_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }
  else if (tid1 == type_f32_rm && tid2 == type_i32_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_rm && tid2 == type_i64_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_rm && tid2 == type_u32_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f32_rm && tid2 == type_u64_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }


  else if (tid1 == type_f64_cm && tid2 == type_i32_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_cm && tid2 == type_i64_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_cm && tid2 == type_u32_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_cm && tid2 == type_u64_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }
  else if (tid1 == type_f64_rm && tid2 == type_i32_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_rm && tid2 == type_i64_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_rm && tid2 == type_u32_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_rm && tid2 == type_u64_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }
  else if (tid1 == type_f64_cm && tid2 == type_i32_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_cm && tid2 == type_i64_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_cm && tid2 == type_u32_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_cm && tid2 == type_u64_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }
  else if (tid1 == type_f64_rm && tid2 == type_i32_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_rm && tid2 == type_i64_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_rm && tid2 == type_u32_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  } else if (tid1 == type_f64_rm && tid2 == type_u64_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> V(v_data, v_shape0, v_shape1);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> F(f_data, f_shape0, f_shape1);
    return std::make_tuple<int, int>(V.rows(), F.rows());
  }

  else {
    cerr << "Type not supported!!!!" << endl;
  }

//  return L.rows();
}


#endif // CRAZY_BINDING1_H

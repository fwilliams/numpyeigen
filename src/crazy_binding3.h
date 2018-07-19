#include <tuple>
#include <iostream>

#include <Eigen/Core>
#include <pybind11/numpy.h>

#include "eigen_typedefs.h"

#ifndef CRAZY_BINDING3_H
#define CRAZY_BINDING3_H

namespace py = pybind11;

std::tuple<int, int, int> craziest_ass_branching_cot_matrix(py::array v, py::array f) {
  using namespace std;

  const char t1 = v.dtype().type();
  const char t2 = f.dtype().type();
  StorateOrder so1 = (v.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (v.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  StorateOrder so2 = (f.flags() & NPY_ARRAY_F_CONTIGUOUS) ? RowMajor : (f.flags() & NPY_ARRAY_C_CONTIGUOUS ? ColMajor : NoContig);
  int tid0 = get_type_id(t1, so1);
  int tid1 = get_type_id(t2, so2);

  long int v_shape[2] = { v.shape()[0], v.shape()[1] };
  long int f_shape[2] = { f.shape()[0], v.shape()[1] };
  long int v_strides[2] = { v.strides()[0], v.strides()[1] };
  long int f_strides[2] = { f.strides()[0], f.strides()[1] };

  if (tid0 == type_f32_cm && tid1 == type_i8_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_cm::Scalar* f_data = (Type_i8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 0);

  } else if (tid0 == type_f32_cm && tid1 == type_i8_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_rm::Scalar* f_data = (Type_i8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 1);

  } else if (tid0 == type_f32_cm && tid1 == type_i8_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_x::Scalar* f_data = (Type_i8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 2);

  } else if (tid0 == type_f32_cm && tid1 == type_i16_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_cm::Scalar* f_data = (Type_i16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 3);

  } else if (tid0 == type_f32_cm && tid1 == type_i16_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_rm::Scalar* f_data = (Type_i16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 4);

  } else if (tid0 == type_f32_cm && tid1 == type_i16_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_x::Scalar* f_data = (Type_i16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 5);

  } else if (tid0 == type_f32_cm && tid1 == type_i32_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 6);

  } else if (tid0 == type_f32_cm && tid1 == type_i32_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 7);

  } else if (tid0 == type_f32_cm && tid1 == type_i32_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_x::Scalar* f_data = (Type_i32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 8);

  } else if (tid0 == type_f32_cm && tid1 == type_i64_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 9);

  } else if (tid0 == type_f32_cm && tid1 == type_i64_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 10);

  } else if (tid0 == type_f32_cm && tid1 == type_i64_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_x::Scalar* f_data = (Type_i64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 11);

  } else if (tid0 == type_f32_cm && tid1 == type_u8_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_cm::Scalar* f_data = (Type_u8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 12);

  } else if (tid0 == type_f32_cm && tid1 == type_u8_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_rm::Scalar* f_data = (Type_u8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 13);

  } else if (tid0 == type_f32_cm && tid1 == type_u8_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_x::Scalar* f_data = (Type_u8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 14);

  } else if (tid0 == type_f32_cm && tid1 == type_u16_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_cm::Scalar* f_data = (Type_u16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 15);

  } else if (tid0 == type_f32_cm && tid1 == type_u16_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_rm::Scalar* f_data = (Type_u16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 16);

  } else if (tid0 == type_f32_cm && tid1 == type_u16_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_x::Scalar* f_data = (Type_u16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 17);

  } else if (tid0 == type_f32_cm && tid1 == type_u32_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 18);

  } else if (tid0 == type_f32_cm && tid1 == type_u32_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 19);

  } else if (tid0 == type_f32_cm && tid1 == type_u32_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_x::Scalar* f_data = (Type_u32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 20);

  } else if (tid0 == type_f32_cm && tid1 == type_u64_cm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 21);

  } else if (tid0 == type_f32_cm && tid1 == type_u64_rm) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 22);

  } else if (tid0 == type_f32_cm && tid1 == type_u64_x) {
    Type_f32_cm::Scalar* v_data = (Type_f32_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_x::Scalar* f_data = (Type_u64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 23);

  } else if (tid0 == type_f32_rm && tid1 == type_i8_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_cm::Scalar* f_data = (Type_i8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 24);

  } else if (tid0 == type_f32_rm && tid1 == type_i8_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_rm::Scalar* f_data = (Type_i8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 25);

  } else if (tid0 == type_f32_rm && tid1 == type_i8_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_x::Scalar* f_data = (Type_i8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 26);

  } else if (tid0 == type_f32_rm && tid1 == type_i16_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_cm::Scalar* f_data = (Type_i16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 27);

  } else if (tid0 == type_f32_rm && tid1 == type_i16_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_rm::Scalar* f_data = (Type_i16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 28);

  } else if (tid0 == type_f32_rm && tid1 == type_i16_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_x::Scalar* f_data = (Type_i16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 29);

  } else if (tid0 == type_f32_rm && tid1 == type_i32_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 30);

  } else if (tid0 == type_f32_rm && tid1 == type_i32_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 31);

  } else if (tid0 == type_f32_rm && tid1 == type_i32_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_x::Scalar* f_data = (Type_i32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 32);

  } else if (tid0 == type_f32_rm && tid1 == type_i64_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 33);

  } else if (tid0 == type_f32_rm && tid1 == type_i64_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 34);

  } else if (tid0 == type_f32_rm && tid1 == type_i64_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_x::Scalar* f_data = (Type_i64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 35);

  } else if (tid0 == type_f32_rm && tid1 == type_u8_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_cm::Scalar* f_data = (Type_u8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 36);

  } else if (tid0 == type_f32_rm && tid1 == type_u8_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_rm::Scalar* f_data = (Type_u8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 37);

  } else if (tid0 == type_f32_rm && tid1 == type_u8_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_x::Scalar* f_data = (Type_u8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 38);

  } else if (tid0 == type_f32_rm && tid1 == type_u16_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_cm::Scalar* f_data = (Type_u16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 39);

  } else if (tid0 == type_f32_rm && tid1 == type_u16_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_rm::Scalar* f_data = (Type_u16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 40);

  } else if (tid0 == type_f32_rm && tid1 == type_u16_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_x::Scalar* f_data = (Type_u16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 41);

  } else if (tid0 == type_f32_rm && tid1 == type_u32_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 42);

  } else if (tid0 == type_f32_rm && tid1 == type_u32_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 43);

  } else if (tid0 == type_f32_rm && tid1 == type_u32_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_x::Scalar* f_data = (Type_u32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 44);

  } else if (tid0 == type_f32_rm && tid1 == type_u64_cm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 45);

  } else if (tid0 == type_f32_rm && tid1 == type_u64_rm) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 46);

  } else if (tid0 == type_f32_rm && tid1 == type_u64_x) {
    Type_f32_rm::Scalar* v_data = (Type_f32_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_x::Scalar* f_data = (Type_u64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 47);

  } else if (tid0 == type_f32_x && tid1 == type_i8_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i8_cm::Scalar* f_data = (Type_i8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 48);

  } else if (tid0 == type_f32_x && tid1 == type_i8_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i8_rm::Scalar* f_data = (Type_i8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 49);

  } else if (tid0 == type_f32_x && tid1 == type_i8_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i8_x::Scalar* f_data = (Type_i8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 50);

  } else if (tid0 == type_f32_x && tid1 == type_i16_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i16_cm::Scalar* f_data = (Type_i16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 51);

  } else if (tid0 == type_f32_x && tid1 == type_i16_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i16_rm::Scalar* f_data = (Type_i16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 52);

  } else if (tid0 == type_f32_x && tid1 == type_i16_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i16_x::Scalar* f_data = (Type_i16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 53);

  } else if (tid0 == type_f32_x && tid1 == type_i32_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 54);

  } else if (tid0 == type_f32_x && tid1 == type_i32_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 55);

  } else if (tid0 == type_f32_x && tid1 == type_i32_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i32_x::Scalar* f_data = (Type_i32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 56);

  } else if (tid0 == type_f32_x && tid1 == type_i64_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 57);

  } else if (tid0 == type_f32_x && tid1 == type_i64_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 58);

  } else if (tid0 == type_f32_x && tid1 == type_i64_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i64_x::Scalar* f_data = (Type_i64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 59);

  } else if (tid0 == type_f32_x && tid1 == type_u8_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u8_cm::Scalar* f_data = (Type_u8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 60);

  } else if (tid0 == type_f32_x && tid1 == type_u8_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u8_rm::Scalar* f_data = (Type_u8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 61);

  } else if (tid0 == type_f32_x && tid1 == type_u8_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u8_x::Scalar* f_data = (Type_u8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 62);

  } else if (tid0 == type_f32_x && tid1 == type_u16_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u16_cm::Scalar* f_data = (Type_u16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 63);

  } else if (tid0 == type_f32_x && tid1 == type_u16_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u16_rm::Scalar* f_data = (Type_u16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 64);

  } else if (tid0 == type_f32_x && tid1 == type_u16_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u16_x::Scalar* f_data = (Type_u16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 65);

  } else if (tid0 == type_f32_x && tid1 == type_u32_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 66);

  } else if (tid0 == type_f32_x && tid1 == type_u32_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 67);

  } else if (tid0 == type_f32_x && tid1 == type_u32_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u32_x::Scalar* f_data = (Type_u32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 68);

  } else if (tid0 == type_f32_x && tid1 == type_u64_cm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 69);

  } else if (tid0 == type_f32_x && tid1 == type_u64_rm) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 70);

  } else if (tid0 == type_f32_x && tid1 == type_u64_x) {
    Type_f32_x::Scalar* v_data = (Type_f32_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u64_x::Scalar* f_data = (Type_u64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 71);

  } else if (tid0 == type_f64_cm && tid1 == type_i8_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_cm::Scalar* f_data = (Type_i8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 72);

  } else if (tid0 == type_f64_cm && tid1 == type_i8_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_rm::Scalar* f_data = (Type_i8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 73);

  } else if (tid0 == type_f64_cm && tid1 == type_i8_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_x::Scalar* f_data = (Type_i8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 74);

  } else if (tid0 == type_f64_cm && tid1 == type_i16_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_cm::Scalar* f_data = (Type_i16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 75);

  } else if (tid0 == type_f64_cm && tid1 == type_i16_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_rm::Scalar* f_data = (Type_i16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 76);

  } else if (tid0 == type_f64_cm && tid1 == type_i16_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_x::Scalar* f_data = (Type_i16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 77);

  } else if (tid0 == type_f64_cm && tid1 == type_i32_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 78);

  } else if (tid0 == type_f64_cm && tid1 == type_i32_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 79);

  } else if (tid0 == type_f64_cm && tid1 == type_i32_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_x::Scalar* f_data = (Type_i32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 80);

  } else if (tid0 == type_f64_cm && tid1 == type_i64_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 81);

  } else if (tid0 == type_f64_cm && tid1 == type_i64_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 82);

  } else if (tid0 == type_f64_cm && tid1 == type_i64_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_x::Scalar* f_data = (Type_i64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 83);

  } else if (tid0 == type_f64_cm && tid1 == type_u8_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_cm::Scalar* f_data = (Type_u8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 84);

  } else if (tid0 == type_f64_cm && tid1 == type_u8_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_rm::Scalar* f_data = (Type_u8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 85);

  } else if (tid0 == type_f64_cm && tid1 == type_u8_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_x::Scalar* f_data = (Type_u8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 86);

  } else if (tid0 == type_f64_cm && tid1 == type_u16_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_cm::Scalar* f_data = (Type_u16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 87);

  } else if (tid0 == type_f64_cm && tid1 == type_u16_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_rm::Scalar* f_data = (Type_u16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 88);

  } else if (tid0 == type_f64_cm && tid1 == type_u16_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_x::Scalar* f_data = (Type_u16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 89);

  } else if (tid0 == type_f64_cm && tid1 == type_u32_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 90);

  } else if (tid0 == type_f64_cm && tid1 == type_u32_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 91);

  } else if (tid0 == type_f64_cm && tid1 == type_u32_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_x::Scalar* f_data = (Type_u32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 92);

  } else if (tid0 == type_f64_cm && tid1 == type_u64_cm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 93);

  } else if (tid0 == type_f64_cm && tid1 == type_u64_rm) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 94);

  } else if (tid0 == type_f64_cm && tid1 == type_u64_x) {
    Type_f64_cm::Scalar* v_data = (Type_f64_cm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_cm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_x::Scalar* f_data = (Type_u64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 95);

  } else if (tid0 == type_f64_rm && tid1 == type_i8_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_cm::Scalar* f_data = (Type_i8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 96);

  } else if (tid0 == type_f64_rm && tid1 == type_i8_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_rm::Scalar* f_data = (Type_i8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 97);

  } else if (tid0 == type_f64_rm && tid1 == type_i8_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i8_x::Scalar* f_data = (Type_i8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 98);

  } else if (tid0 == type_f64_rm && tid1 == type_i16_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_cm::Scalar* f_data = (Type_i16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 99);

  } else if (tid0 == type_f64_rm && tid1 == type_i16_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_rm::Scalar* f_data = (Type_i16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 100);

  } else if (tid0 == type_f64_rm && tid1 == type_i16_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i16_x::Scalar* f_data = (Type_i16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 101);

  } else if (tid0 == type_f64_rm && tid1 == type_i32_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 102);

  } else if (tid0 == type_f64_rm && tid1 == type_i32_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 103);

  } else if (tid0 == type_f64_rm && tid1 == type_i32_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i32_x::Scalar* f_data = (Type_i32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 104);

  } else if (tid0 == type_f64_rm && tid1 == type_i64_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 105);

  } else if (tid0 == type_f64_rm && tid1 == type_i64_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 106);

  } else if (tid0 == type_f64_rm && tid1 == type_i64_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_i64_x::Scalar* f_data = (Type_i64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 107);

  } else if (tid0 == type_f64_rm && tid1 == type_u8_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_cm::Scalar* f_data = (Type_u8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 108);

  } else if (tid0 == type_f64_rm && tid1 == type_u8_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_rm::Scalar* f_data = (Type_u8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 109);

  } else if (tid0 == type_f64_rm && tid1 == type_u8_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u8_x::Scalar* f_data = (Type_u8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 110);

  } else if (tid0 == type_f64_rm && tid1 == type_u16_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_cm::Scalar* f_data = (Type_u16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 111);

  } else if (tid0 == type_f64_rm && tid1 == type_u16_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_rm::Scalar* f_data = (Type_u16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 112);

  } else if (tid0 == type_f64_rm && tid1 == type_u16_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u16_x::Scalar* f_data = (Type_u16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 113);

  } else if (tid0 == type_f64_rm && tid1 == type_u32_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 114);

  } else if (tid0 == type_f64_rm && tid1 == type_u32_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 115);

  } else if (tid0 == type_f64_rm && tid1 == type_u32_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u32_x::Scalar* f_data = (Type_u32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 116);

  } else if (tid0 == type_f64_rm && tid1 == type_u64_cm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 117);

  } else if (tid0 == type_f64_rm && tid1 == type_u64_rm) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 118);

  } else if (tid0 == type_f64_rm && tid1 == type_u64_x) {
    Type_f64_rm::Scalar* v_data = (Type_f64_rm::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_rm, Eigen::Aligned> v_eigen(v_data, v_shape[0], v_shape[1]);
    Type_u64_x::Scalar* f_data = (Type_u64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 119);

  } else if (tid0 == type_f64_x && tid1 == type_i8_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i8_cm::Scalar* f_data = (Type_i8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 120);

  } else if (tid0 == type_f64_x && tid1 == type_i8_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i8_rm::Scalar* f_data = (Type_i8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 121);

  } else if (tid0 == type_f64_x && tid1 == type_i8_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i8_x::Scalar* f_data = (Type_i8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 122);

  } else if (tid0 == type_f64_x && tid1 == type_i16_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i16_cm::Scalar* f_data = (Type_i16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 123);

  } else if (tid0 == type_f64_x && tid1 == type_i16_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i16_rm::Scalar* f_data = (Type_i16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 124);

  } else if (tid0 == type_f64_x && tid1 == type_i16_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i16_x::Scalar* f_data = (Type_i16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 125);

  } else if (tid0 == type_f64_x && tid1 == type_i32_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i32_cm::Scalar* f_data = (Type_i32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 126);

  } else if (tid0 == type_f64_x && tid1 == type_i32_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i32_rm::Scalar* f_data = (Type_i32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 127);

  } else if (tid0 == type_f64_x && tid1 == type_i32_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i32_x::Scalar* f_data = (Type_i32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 128);

  } else if (tid0 == type_f64_x && tid1 == type_i64_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i64_cm::Scalar* f_data = (Type_i64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 129);

  } else if (tid0 == type_f64_x && tid1 == type_i64_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i64_rm::Scalar* f_data = (Type_i64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 130);

  } else if (tid0 == type_f64_x && tid1 == type_i64_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_i64_x::Scalar* f_data = (Type_i64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_i64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 131);

  } else if (tid0 == type_f64_x && tid1 == type_u8_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u8_cm::Scalar* f_data = (Type_u8_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 132);

  } else if (tid0 == type_f64_x && tid1 == type_u8_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u8_rm::Scalar* f_data = (Type_u8_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 133);

  } else if (tid0 == type_f64_x && tid1 == type_u8_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u8_x::Scalar* f_data = (Type_u8_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u8_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 134);

  } else if (tid0 == type_f64_x && tid1 == type_u16_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u16_cm::Scalar* f_data = (Type_u16_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 135);

  } else if (tid0 == type_f64_x && tid1 == type_u16_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u16_rm::Scalar* f_data = (Type_u16_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 136);

  } else if (tid0 == type_f64_x && tid1 == type_u16_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u16_x::Scalar* f_data = (Type_u16_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u16_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 137);

  } else if (tid0 == type_f64_x && tid1 == type_u32_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u32_cm::Scalar* f_data = (Type_u32_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 138);

  } else if (tid0 == type_f64_x && tid1 == type_u32_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u32_rm::Scalar* f_data = (Type_u32_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 139);

  } else if (tid0 == type_f64_x && tid1 == type_u32_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u32_x::Scalar* f_data = (Type_u32_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u32_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 140);

  } else if (tid0 == type_f64_x && tid1 == type_u64_cm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u64_cm::Scalar* f_data = (Type_u64_cm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_cm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 141);

  } else if (tid0 == type_f64_x && tid1 == type_u64_rm) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u64_rm::Scalar* f_data = (Type_u64_rm::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_rm, Eigen::Aligned> f_eigen(f_data, f_shape[0], f_shape[1]);
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 142);

  } else if (tid0 == type_f64_x && tid1 == type_u64_x) {
    Type_f64_x::Scalar* v_data = (Type_f64_x::Scalar*)(v.data(0));
    Eigen::Map<Type_f64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        v_eigen(v_data, v_shape[0], v_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(v_strides[0], v_strides[1]));
    Type_u64_x::Scalar* f_data = (Type_u64_x::Scalar*)(f.data(0));
    Eigen::Map<Type_u64_x, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        f_eigen(f_data, f_shape[0], f_shape[1],
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(f_strides[0], f_strides[1]));
    return std::make_tuple<int, int, int>(v_eigen.rows(), f_eigen.rows(), 143);

  } else {
    cerr << "Type not supported!!!!" << endl;
    cerr << tid0 << " " << tid1 << endl;
  }

  //  return L.rows();
}


#endif // CRAZY_BINDING3_H

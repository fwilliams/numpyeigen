#include <Eigen/Core>

igl_binding("test_binding")
igl_input("b", "matches(a)")
igl_input("a", "type_f32", "type_f64")
igl_input("c", "type_i32", "type_i64")
igl_input("d", "matches(c)")
igl_input("e", "matches(d)")
igl_begin_code()

typedef IGL_PY_TYPE_a::Scalar Scalar_a;
typedef IGL_PY_TYPE_a::Eigen_Type Matrix_a;

typedef IGL_PY_TYPE_b::Scalar Scalar_b;
typedef IGL_PY_TYPE_b::Eigen_Type Matrix_b;

Scalar_a* a_data = (Scalar_a*) a.data();
int a_shape_0 = a.shape()[0];
int a_shape_1 = a.shape()[1];
Eigen::Map<Matrix_a, IGL_PY_TYPE_a::Aligned> A(a_data, a_shape_0, a_shape_1);

Scalar_b* b_data = (Scalar_b*) b.data();
int b_shape_0 = b.shape()[0];
int b_shape_1 = b.shape()[1];
Eigen::Map<Matrix_b, IGL_PY_TYPE_b::Aligned> B(b_data, b_shape_0, b_shape_1);

Matrix_a C = A + B;

// TODO: Check that this is doing a move and not a copy
return pybind11::detail::eigen_encapsulate<pybind11::detail::EigenProps<Matrix_a>>(new Matrix_a(std::move(C)));

igl_end_code()



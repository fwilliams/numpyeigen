npe_function("mutate_matrix")
npe_arg("a", "type_f64")
npe_begin_code()

typedef NPE_PY_TYPE_a::Scalar Scalar_a;
typedef NPE_PY_TYPE_a::Eigen_Type Matrix_a;
typedef NPE_PY_TYPE_a::Map_Type Map_a;

Map_a A((Scalar_a*)a.data(), a.shape()[0], a.shape()[1]);
A(0, 0) = 2.0;

return A.rows() + A.cols();

npe_end_code()

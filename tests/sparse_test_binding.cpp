#include <pybind11/pybind11.h>
#include <sparse_array.h>

PYBIND11_MODULE(sparse_test, m) {
  m.doc() = "used for sparse_test.py";

  m.def("test", [](npe::sparse_array sp) {
    std::string fmt = sp.getformat();

    std::cout << sp.indices().shape()[0] << " " << sp.data().shape()[0] << " " << fmt << std::endl;
    std::cout << sp.data().dtype().type() << std::endl;

    return sp;
  });
}

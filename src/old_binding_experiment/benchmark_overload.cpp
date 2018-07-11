#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;


PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";


    m.def("overloaded0", [](py::array_t<double>, py::array_t<double>) { return "double"; });
    m.def("overloaded0", [](py::array_t<float>, py::array_t<float>) { return "double"; });


    m.def("overloaded1", [](py::array_t<double>, py::array_t<double>) { return "double"; });
    m.def("overloaded1", [](py::array_t<float>, py::array_t<double>) { return "float"; });
    m.def("overloaded1", [](py::array_t<int>, py::array_t<double>) { return "int"; });
    m.def("overloaded1", [](py::array_t<unsigned short>, py::array_t<double>) { return "unsigned short"; });
    m.def("overloaded1", [](py::array_t<std::uint64_t>, py::array_t<double>) { return "std::uint64_t"; });
    m.def("overloaded1", [](py::array_t<std::complex<double>>, py::array_t<double>) { return "double complex"; });
    m.def("overloaded1", [](py::array_t<std::complex<float>>, py::array_t<double>) { return "float complex"; });


    m.def("overloaded2", [](py::array_t<double>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<double>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<double>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<double>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<double>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<double>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<double>, py::array_t<std::complex<float>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<float>, py::array_t<std::complex<float>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<int>, py::array_t<std::complex<float>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<unsigned short>, py::array_t<std::complex<float>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::uint64_t>, py::array_t<std::complex<float>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<double>>, py::array_t<std::complex<float>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<double>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<float>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<int>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<unsigned  short>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<std::uint64_t>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<std::complex<double>>) { return "double"; });
    m.def("overloaded2", [](py::array_t<std::complex<float>>, py::array_t<std::complex<float>>) { return "double"; });


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

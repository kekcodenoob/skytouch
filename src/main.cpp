#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

// The module name MUST match the target name in CMakeLists.txt (_core)
PYBIND11_MODULE(_core, m) {
    m.doc() = "SkyTouch C++ Core Plugin";
    m.def("add", &add, "A function that adds two numbers", py::arg("a"), py::arg("b"));
}
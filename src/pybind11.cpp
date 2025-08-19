// Project-specific includes
#include "kspt.hpp"

// Pybind11 includes
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


PYBIND11_MODULE(_kspt, m) {
    m.def(
        "rand_max_split_ks",
        &rand_max_split_ks,
        py::arg("data"),
        py::arg("num_samples"),
        py::arg("num_bins"),
        py::arg("min_split_size"),
        py::arg("coarse_scan_width"),
        py::arg("seed")
    );
}
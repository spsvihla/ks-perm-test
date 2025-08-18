/**
 * @file kspt.hpp
 * @brief 
 * @author Sean Svihla
 */
#ifndef KSPT_HPP
#define KSPT_HPP

// Pybind11 includes
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * @brief
 * @param seq
 * @param min_split_size
 * @param num_bins
 * @param num_samples
 * @return 
 */
py::array_t<double> rand_max_split_ks(py::array_t<double> data, int num_samples, 
                                      int num_bins, int min_split_size, 
                                      int coarse_scan_width, 
                                      std::optional<unsigned int> seed);

#endif // KSPT_HPP
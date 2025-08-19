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
 * @brief Compute randomized maximum-split Kolmogorov–Smirnov distances
 *        from a given data sequence.
 *
 * This function repeatedly shuffles the input data and computes the
 * maximum KS distance over all possible splits. A coarse scan is performed
 * first to identify promising split regions, followed by a refinement step
 * around the best coarse split. The resulting distribution of maximum KS
 * distances is returned as a NumPy array.
 *
 * @param data A 1D NumPy array of doubles representing the input sequence.
 * @param num_samples The number of randomized shuffles / KS computations to perform.
 * @param num_bins The number of bins to use for histogram-based empirical CDFs.
 * @param min_split_size The minimum allowable size for a split (both left and right).
 * @param coarse_scan_width The step size for the initial coarse scan of possible splits.
 * @param seed Optional random seed for reproducibility; if not provided, a random device is used.
 * @return The observed value from the original ordering of data, and a
 *         1D NumPy array of doubles of length `num_samples`, containing the
 *         maximum KS distance for each randomized shuffle of the input data.
 */
py::tuple rand_max_split_ks(py::array_t<double> data, int num_samples, 
                            int num_bins, int min_split_size, 
                            int coarse_scan_width, 
                            std::optional<unsigned int> seed);

#endif // KSPT_HPP
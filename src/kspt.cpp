/**
 * @file kspt.cpp
 * @author Sean Svihla
 */

// Standard library includes
#include <algorithm>
#include <cmath>
#include <tuple>
#include <random>
#include <vector>

// Project-specific includes
#include "kspt.hpp"

// Pybind11 includes
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


struct HistBins {
    std::vector<double> edges;
    double min_val;
    double max_val;
    double bin_width;
    int num_bins;
};


// make 'num_bins' bins for data
inline HistBins
make_bins(double* data, int size, int num_bins)
{
    double min_val = *std::min_element(data, data + size);
    double max_val = *std::max_element(data, data + size);
    double bin_width = (max_val - min_val) / num_bins;

    std::vector<double> edges(num_bins + 1);
    for(int i = 0; i <= num_bins; ++i)
    {
        edges[i] = min_val + i * bin_width;
    }

    return HistBins{edges, min_val, max_val, bin_width, num_bins};
}

// in-place Fisher-Yates shuffle
inline void
fisher_yates(double* arr, int size, std::mt19937& rng)
{
    // this loop might be hard to optimize -- so unroll it manually
    std::size_t max = rng.max();
    std::size_t j = size - 1;
    for(; j + 1 >= 4; j -= 4) 
    {
        // iteration 1
        std::size_t k1;
        do 
        {
            k1 = rng();
        } while (k1 >= max - (max % (j + 1)));
        k1 %= (j + 1);
        std::swap(arr[j], arr[k1]);

        // iteration 2
        std::size_t k2;
        do 
        {
            k2 = rng();
        } while (k2 >= max - (max % (j)));
        k2 %= j;
        std::swap(arr[j - 1], arr[k2]);

        // iteration 3
        std::size_t k3;
        do 
        {
            k3 = rng();
        } while (k3 >= max - (max % (j - 1)));
        k3 %= (j - 1);
        std::swap(arr[j - 2], arr[k3]);

        // iteration 4
        std::size_t k4;
        do 
        {
            k4 = rng();
        } while (k4 >= max - (max % (j - 2)));
        k4 %= (j - 2);
        std::swap(arr[j - 3], arr[k4]);
    }

    // Remainder loop for leftovers
    for(; j > 0 && j != std::numeric_limits<size_t>::max(); --j) 
    {
        std::size_t k;
        do 
        {
            k = rng();
        } while (k >= max - (max % (j + 1)));
        k %= (j + 1);
        std::swap(arr[j], arr[k]);
    }
}

inline std::vector<int>
make_coarse_splits(int size, int min_split_size, int width)
{
    std::vector<int> splits;
    splits.reserve((size - 2 * min_split_size) / width);
    for(int i = min_split_size; i < size - min_split_size; i+= width)
    {
        splits.push_back(i);
    }
    return splits;
}

inline std::vector<int>
make_refinement_splits(const std::vector<int>& coarse_splits, int jmax,
                       int min_split_size, int size)
{
    int start = (jmax == 0) ? min_split_size : coarse_splits[jmax - 1] + 1;
    int end = (jmax + 1 >= coarse_splits.size()) ? size - min_split_size : coarse_splits[jmax + 1] - 1;

    std::vector<int> fine_splits;
    for(int k = start; k <= end; ++k) 
    {
        fine_splits.push_back(k);
    }

    return fine_splits;
}

// prefix cumulative histogram
inline std::vector<std::vector<int>>
make_prefix_cumhist(double* data, int size,
                    const HistBins& bins, const std::vector<int>& splits)
{
    std::vector<std::vector<int>> cumhists;
    cumhists.reserve(splits.size() + 1);

    std::vector<int> hist(bins.num_bins, 0);
    int next = 0;

    for(int i = 0; i < size; ++i)
    {
        if(i == splits[next])
        {
            std::vector<int> cumhist(hist.size());
            std::partial_sum(hist.begin(), hist.end(), cumhist.begin());
            cumhists.push_back(cumhist);
            next++;
        }
        int bin = static_cast<int>((data[i] - bins.min_val) / bins.bin_width);
        bin = std::min(bins.num_bins - 1, bin);
        hist[bin]++;
    }

    // append total counts
    std::vector<int> cumhist(hist.size());
    std::partial_sum(hist.begin(), hist.end(), cumhist.begin());
    cumhists.push_back(cumhist);

    return cumhists;
}

inline double
F_right_b(const std::vector<int>& splits,
          const std::vector<std::vector<int>>& cumhists,
          int j, int b, int size)
{
    return static_cast<double>(cumhists.back()[b] - cumhists[j][b]) / (size - splits[j]);
}

inline double 
F_left_b(const std::vector<int>& splits, 
         const std::vector<std::vector<int>>& cumhists,
         int j, int b)
{
    return static_cast<double>(cumhists[j][b]) / splits[j];
}

// compute max KS distance over splits
inline std::tuple<double, int>
compute_max_split_ks(const std::vector<int>& splits,
                     const std::vector<std::vector<int>>& cumhists,
                     int size, int num_bins, int num_splits)
{
    double maxmax = 0;
    int jmax = -1;
    for(int j = 0; j < num_splits; ++j)
    {
        double max = 0;
        for(int b = 0; b < num_bins; ++b)
        {
            double val = std::abs(
                F_right_b(splits, cumhists, j, b, size) - 
                F_left_b(splits, cumhists, j, b)
            );
            max = std::max(max, val);
        }
        if(max > maxmax)
        {
            maxmax = max;
            jmax = j;
        }
    }
    return std::make_tuple(maxmax, jmax);
}

// find max-split KS distance
inline double
max_split_ks_dist(double* data, int size, const HistBins& bins,
                  int min_split_size, int coarse_scan_width)
{
    // coarse scan
    std::vector<int> c_splits = make_coarse_splits(size, min_split_size, coarse_scan_width);
    std::vector<std::vector<int>> c_cumhists = make_prefix_cumhist(data, size, bins, c_splits);
    auto [c_maxmax, c_jmax] = compute_max_split_ks(c_splits, c_cumhists, size, bins.num_bins, c_splits.size());

    // refinement
    std::vector<int> r_splits = make_refinement_splits(c_splits, c_jmax, min_split_size, size);
    std::vector<std::vector<int>> r_cumhists = make_prefix_cumhist(data, size, bins, r_splits);
    auto [r_maxmax, r_jmax] = compute_max_split_ks(r_splits, r_cumhists, size, bins.num_bins, r_splits.size());

    return r_maxmax;
}

py::tuple
rand_max_split_ks(py::array_t<double> data, int num_samples, int num_bins,
                  int min_split_size, int coarse_scan_width, 
                  std::optional<unsigned int> seed)
{
    double* data_ = static_cast<double*>(data.request().ptr);
    int size = static_cast<int>(data.size());

    unsigned int seed_ = seed.value_or(std::random_device{}());
    std::mt19937 rng(seed_);

    HistBins bins = make_bins(data_, size, num_bins);

    py::array_t<double> samples(num_samples);
    double* samples_ = static_cast<double*>(samples.request().ptr);

    // observed value
    double obs = max_split_ks_dist(data_, size, bins, min_split_size, 
                                   coarse_scan_width);

    // TODO: parallelize loop
    // samples for estimating cdf
    for(int i = 0; i < num_samples; ++i)
    {
        fisher_yates(data_, size, rng);
        samples_[i] = max_split_ks_dist(data_, size, bins, min_split_size, 
                                        coarse_scan_width);
    }

    return py::make_tuple(obs, samples);
}

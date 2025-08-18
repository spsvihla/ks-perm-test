/**
 * @file kspt.cpp
 * @author Sean Svihla
 */

// Standard library includes
#include <random>

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
} typedef Struct;


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

// Kolmogorov-Smirnov distance
inline double
ks_dist(std::vector<double> left, std::vector<double> right)
{
    double max = 0;
    for(std::size_t i = 0; i < left.size(); ++i)
    {
        double d = std::abs(left[i] - right[i]);
        max = std::max(max, d);
    }
    return max;
}

// empirical distrbution function over bins
inline std::vector<double>
make_ecdf(double* data, int size, HistBins& bins)
{
    std::vector<double> ecdf(bins.num_bins, 0);

    // compute counts
    for(int i = 0; i < size; ++i)
    {
        int bin = static_cast<int>((data[i] - bins.min_val) / bins.bin_width);
        bin = std::min(bins.num_bins - 1, bin);
        ecdf[bin]++;
    }

    // compute cumsum in-place
    int cumsum = 0;
    for(int i = 0; i < bins.num_bins; ++i)
    {
        cumsum += ecdf[i];
        ecdf[i] = static_cast<double>(cumsum / size);
    }

    return ecdf;
}

// find max-split KS distance
inline double
max_split_ks_dist(double* data, int size, HistBins& bins, int min_split_size, 
                  int coarse_scan_width)
{

}

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

py::array_t<double>
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

    for(int i = 0; i < num_samples; ++i)
    {
        fisher_yates(data_, size, rng);
        samples_[i] = max_split_ks_dist(data_, size, bins, min_split_size, 
                                        coarse_scan_width);
    }

    return samples;
}

#pragma once

#include <complex>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

class Functions
{
public:
    //  real -> complex
    static std::vector<std::complex<float>> fft(const float * data, size_t data_size);

    // static std::vector<std::complex<float>> fftWithCompress(const std::vector<float> & data);
};

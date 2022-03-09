/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <string.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

template <typename T>
std::vector<T> genRandomWithSizeAndMaxValue(size_t size, size_t max_value = -1)
{
    std::vector<T> vec;
    vec.reserve(size);
    srand(time(NULL));
    for (size_t i = 0; i < size; ++i)
    {
        vec.emplace_back(rand() % max_value + 1);
    }
    return vec;
}

size_t compute_batch_size(const std::vector<std::vector<char>> & data, const size_t chunk_size);

size_t compute_batch_size_with_data_size(const size_t uncompressed_data_size, const size_t chunk_size);

std::vector<size_t>
compute_chunk_sizes_with_data_size(const size_t uncompressed_data_size, const size_t batch_size, const size_t chunk_size);

std::vector<size_t> compute_chunk_sizes(const std::vector<std::vector<char>> & data, const size_t batch_size, const size_t chunk_size);

std::vector<void *> get_input_ptrs_with_row_data(const char * data, const size_t batch_size, const size_t chunk_size);

std::vector<void *> get_input_ptrs(const std::vector<std::vector<char>> & data, const size_t batch_size, const size_t chunk_size);

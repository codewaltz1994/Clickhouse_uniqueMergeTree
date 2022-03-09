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
#include <Stopwatch.h>
#include "BatchDataCPU.h"

class BatchData
{
public:
    BatchData(const char * uncompressed_data, const size_t uncompressed_data_size, const size_t chunk_size)
        : m_ptrs(), m_sizes(), m_data(), m_size()
    {
        m_size = compute_batch_size_with_data_size(uncompressed_data_size, chunk_size);
        m_data = thrust::device_vector<uint8_t>(chunk_size * size());

        std::vector<void *> uncompressed_ptrs(size());
        for (size_t i = 0; i < size(); ++i)
        {
            uncompressed_ptrs[i] = static_cast<void *>(data() + chunk_size * i);
        }
        m_ptrs = thrust::device_vector<void *>(uncompressed_ptrs);
        std::vector<size_t> sizes = compute_chunk_sizes_with_data_size(uncompressed_data_size, size(), chunk_size);
        m_sizes = thrust::device_vector<size_t>(sizes);
        CUDA_CHECK(cudaMemcpy(uncompressed_ptrs[0], uncompressed_data, uncompressed_data_size, cudaMemcpyHostToDevice));
    }

    BatchData(const std::vector<std::vector<char>> & host_data, const size_t chunk_size) : m_ptrs(), m_sizes(), m_data(), m_size(0)
    {
        m_size = compute_batch_size(host_data, chunk_size);

        m_data = thrust::device_vector<uint8_t>(chunk_size * size());

        std::vector<void *> uncompressed_ptrs(size());
        for (size_t i = 0; i < size(); ++i)
        {
            uncompressed_ptrs[i] = static_cast<void *>(data() + chunk_size * i);
        }

        m_ptrs = thrust::device_vector<void *>(uncompressed_ptrs);
        std::vector<size_t> sizes = compute_chunk_sizes(host_data, size(), chunk_size);
        m_sizes = thrust::device_vector<size_t>(sizes);

        // copy data to GPU
        size_t offset = 0;
        for (size_t i = 0; i < host_data.size(); ++i)
        {
            CUDA_CHECK(cudaMemcpy(uncompressed_ptrs[offset], host_data[i].data(), host_data[i].size(), cudaMemcpyHostToDevice));

            const size_t num_chunks = (host_data[i].size() + chunk_size - 1) / chunk_size;
            offset += num_chunks;
        }
  }

  void compact()
  {
      size_t offset = sizes()[0];
      for (size_t i = 0; i < size(); ++i)
      {
          cudaMemcpy(data() + offset, ptrs()[i], sizes()[i], cudaMemcpyDeviceToDevice);
          offset += sizes()[i];
      }
      m_data.resize(offset);
  }

  BatchData(const BatchDataCPU & batch_data, bool copy_data = false, bool compacted = false) : m_ptrs(), m_sizes(), m_data(), m_size()
  {
      m_size = batch_data.size();
      m_sizes = thrust::device_vector<size_t>(batch_data.sizes(), batch_data.sizes() + size());

      size_t data_size
          = compacted ? batch_data.data_size() : std::accumulate(batch_data.sizes(), batch_data.sizes() + size(), static_cast<size_t>(0));
      m_data = thrust::device_vector<uint8_t>(data_size);

      size_t offset = 0;
      std::vector<void *> ptrs(size());
      for (size_t i = 0; i < size(); ++i)
      {
          ptrs[i] = data() + offset;
          offset += batch_data.sizes()[i];
      }
      m_ptrs = thrust::device_vector<void *>(ptrs);

      if (copy_data)
      {
          if (compacted)
          {
              CUDA_CHECK(cudaMemcpy(data(), batch_data.data(), data_size, cudaMemcpyHostToDevice));
          }
          else
          {
              const void * const * src = batch_data.ptrs();
              const size_t * bytes = batch_data.sizes();
              for (size_t i = 0; i < size(); ++i)
              {
                  CUDA_CHECK(cudaMemcpy(ptrs[i], src[i], bytes[i], cudaMemcpyHostToDevice));
              }
          }
      }
  }

  BatchData(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(batch_size)
  {
    m_data = thrust::device_vector<uint8_t>(max_output_size * size());

    std::vector<size_t> sizes(size(), max_output_size);
    m_sizes = thrust::device_vector<size_t>(sizes);

    std::vector<void*> ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      ptrs[i] = data() + max_output_size * i;
    }
    m_ptrs = thrust::device_vector<void*>(ptrs);
  }

  BatchData(BatchData&& other) = default;

  // disable copying
  BatchData(const BatchData& other) = delete;
  BatchData& operator=(const BatchData& other) = delete;

  uint8_t* data()
  {
    return m_data.data().get();
  }
  const uint8_t* data() const
  {
    return m_data.data().get();
  }

  void** ptrs()
  {
    return m_ptrs.data().get();
  }
  const void* const* ptrs() const
  {
    return m_ptrs.data().get();
  }

  size_t* sizes()
  {
    return m_sizes.data().get();
  }
  const size_t* sizes() const
  {
    return m_sizes.data().get();
  }

  size_t size() const
  {
    return m_size;
  }

  size_t data_size() const { return m_data.size(); }

  private:
  thrust::device_vector<void*> m_ptrs;
  thrust::device_vector<size_t> m_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

inline bool operator==(const BatchDataCPU& lhs, const BatchData& rhs)
{
  size_t batch_size = lhs.size();

  if (lhs.size() != rhs.size())
    return false;

  std::vector<size_t> rhs_sizes(rhs.size());
  CUDA_CHECK(cudaMemcpy(
      rhs_sizes.data(),
      rhs.sizes(),
      rhs.size() * sizeof(size_t),
      cudaMemcpyDeviceToHost));

  std::vector<void*> rhs_ptrs(rhs.size());
  CUDA_CHECK(cudaMemcpy(
      rhs_ptrs.data(),
      rhs.ptrs(),
      rhs.size() * sizeof(void*),
      cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < batch_size; ++i) {
    if (lhs.sizes()[i] != rhs_sizes[i])
      return false;

    const uint8_t* lhs_ptr = reinterpret_cast<const uint8_t*>(lhs.ptrs()[i]);
    const uint8_t* rhs_ptr = reinterpret_cast<const uint8_t*>(rhs_ptrs[i]);
    std::vector<uint8_t> rhs_data(rhs_sizes[i]);
    CUDA_CHECK(cudaMemcpy(
        rhs_data.data(), rhs_ptr, rhs_sizes[i], cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < rhs_sizes[i]; ++j)
      if (lhs_ptr[j] != rhs_data[j]) {
        return false;
      }
  }
  return true;
}

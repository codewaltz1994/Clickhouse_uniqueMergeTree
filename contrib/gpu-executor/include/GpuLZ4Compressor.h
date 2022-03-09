#pragma once

#include <iostream>
#include <BatchData.h>
#include <assert.h>
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp.hpp>

using namespace nvcomp;

namespace GpuLZ4
{

size_t compress(uint8_t * device_input_ptrs, size_t input_buffer_len, uint8_t *& dst);

void decompress(uint8_t * device_input_ptrs, uint8_t *& dst);

BatchData compress(const BatchData & input_data);

BatchData decompress(const BatchData & compress_data, BatchDataCPU & input_data_cpu);
}

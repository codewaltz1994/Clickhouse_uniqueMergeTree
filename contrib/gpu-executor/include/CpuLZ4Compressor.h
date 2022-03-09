#pragma once

#include <BatchData.h>
#include <BatchDataCPU.h>
#include <lz4.h>
#include <lz4hc.h>

namespace CpuLZ4
{
/// dst must be allocated
size_t getMaxCompressedDataSize(int uncompressed_size);
size_t compress(const char * src, size_t src_size, char * dst);
/// dst must be allocated
size_t decompress(const char * src, size_t src_size, char * dst, size_t uncompressed_size);

BatchDataCPU GetBatchDataCPU(const BatchData & batch_data, bool copy_data, bool compacted = false);

BatchDataCPU compress(const BatchDataCPU & input_data_cpu);

BatchDataCPU decompress(const BatchDataCPU & compress_data_cpu, const BatchData & input_data);
BatchDataCPU decompress(const BatchDataCPU & compress_data_cpu, size_t uncompressed_size);
}

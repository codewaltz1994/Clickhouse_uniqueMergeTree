#include <define.h>
#include <GpuLZ4Compressor.h>

namespace GpuLZ4
{
/// dst does not need to allocated before call this func, but the caller should responsible for free it after use.
/// high level API, the compressed buffer only can decompressed by nvcomp high level API
size_t compress(uint8_t * device_input_ptrs, size_t input_buffer_len, uint8_t *& dst)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    nvcompType_t data_type = NVCOMP_TYPE_CHAR;

    LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

    CUDA_CHECK(cudaMalloc(&dst, comp_config.max_compressed_buffer_size));

    nvcomp_manager.compress(device_input_ptrs, dst, comp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));

    return nvcomp_manager.get_compressed_output_size(dst);
}

/// the device_input_ptrs must be buffer compressed by nvcomp high level api
void decompress(uint8_t * device_input_ptrs, uint8_t *& dst)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    nvcompType_t data_type = NVCOMP_TYPE_CHAR;

    LZ4Manager nvcomp_manager{chunk_size, data_type, stream};

    DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(device_input_ptrs);

    CUDA_CHECK(cudaMalloc(&dst, decomp_config.decomp_data_size));

    nvcomp_manager.decompress(dst, device_input_ptrs, decomp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

BatchData compress(const BatchData & input_data)
{
    // Compress on the GPU using batched API
    size_t comp_temp_bytes;
    nvcompStatus_t status
        = nvcompBatchedLZ4CompressGetTempSize(input_data.size(), chunk_size, nvcompBatchedLZ4DefaultOpts, &comp_temp_bytes);
    if (status != nvcompSuccess)
    {
        throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetTempSize() not successful");
    }

    void * d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    size_t max_out_bytes;
    status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);
    if (status != nvcompSuccess)
    {
        throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetMaxOutputChunkSize() not successful");
    }

    BatchData compress_data(max_out_bytes, input_data.size());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);

    status = nvcompBatchedLZ4CompressAsync(
        input_data.ptrs(),
        input_data.sizes(),
        chunk_size,
        input_data.size(),
        d_comp_temp,
        comp_temp_bytes,
        compress_data.ptrs(),
        compress_data.sizes(),
        nvcompBatchedLZ4DefaultOpts,
        stream);
    if (status != nvcompSuccess)
    {
        throw std::runtime_error("nvcompBatchedLZ4CompressAsync() failed.");
    }

    cudaEventRecord(end, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // free compression memory
    cudaFree(d_comp_temp);

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    // compute compression ratio
    std::vector<size_t> compressed_sizes_host(compress_data.size());
    cudaMemcpy(
        compressed_sizes_host.data(), compress_data.sizes(), compress_data.size() * sizeof(*compress_data.sizes()), cudaMemcpyDeviceToHost);

    size_t comp_bytes = 0;
    for (const size_t s : compressed_sizes_host)
    {
        comp_bytes += s;
    }

    std::cout << "Compressed ratio : " << (double)input_data.data_size() / comp_bytes << std::endl;
    std::cout << "GPU compression time: " << ms << " ms" << std::endl;
    std::cout << "GPU compression throughput (GB/s): " << (double)input_data.data_size() / (1.0e6 * ms) << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    return compress_data;
}

BatchData decompress(const BatchData & compress_data, BatchDataCPU & input_data_cpu)
{
    // Allocate and build up decompression batch on GPU
    BatchData decomp_data(input_data_cpu, false, false);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // CUDA events to measure decompression time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // lz4 GPU decompression
    size_t decomp_temp_bytes;
    nvcompStatus_t status = nvcompBatchedLZ4DecompressGetTempSize(compress_data.size(), chunk_size, &decomp_temp_bytes);
    if (status != nvcompSuccess)
    {
        throw std::runtime_error("nvcompBatchedLZ4DecompressGetTempSize() failed.");
    }

    void * d_decomp_temp;
    CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

    size_t * d_decomp_sizes;
    CUDA_CHECK(cudaMalloc((void **)&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));

    nvcompStatus_t * d_status_ptrs;
    CUDA_CHECK(cudaMalloc((void **)&d_status_ptrs, decomp_data.size() * sizeof(nvcompStatus_t)));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run decompression
    cudaEventRecord(start, stream);
    status = nvcompBatchedLZ4DecompressAsync(
        compress_data.ptrs(),
        compress_data.sizes(),
        decomp_data.sizes(),
        d_decomp_sizes,
        compress_data.size(),
        d_decomp_temp,
        decomp_temp_bytes,
        decomp_data.ptrs(),
        d_status_ptrs,
        stream);
    cudaEventRecord(end, stream);
    if (status != nvcompSuccess)
    {
        throw std::runtime_error("ERROR: nvcompBatchedLZ4DecompressAsync() not successful");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    double decompression_throughput = ((double)compress_data.data_size() / ms) * 1e-6;
    std::cout << "GPU decompression time: " << ms << " ms" << std::endl;
    std::cout << "GPU decompression throughput (GB/s): " << decompression_throughput << std::endl << std::endl;

    cudaFree(d_decomp_temp);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    return decomp_data;
}
}

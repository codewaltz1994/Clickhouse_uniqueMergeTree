#include <atomic>
#include <thread>
#include <Stopwatch.h>
#include <define.h>
#include <getNumberOfPhysicalCPUCores.h>
#include <CpuLZ4Compressor.h>


namespace CpuLZ4
{
size_t getMaxCompressedDataSize(int uncompressed_size)
{
    return LZ4_COMPRESSBOUND(uncompressed_size);
}
size_t compress(const char * src, size_t src_size, char * dst)
{
    return LZ4_compress_default(src, dst, src_size, getMaxCompressedDataSize(src_size));
}
size_t decompress(const char * src, size_t src_size, char * dst, size_t uncompressed_size)
{
    return LZ4_decompress_safe(src, dst, src_size, uncompressed_size);
}

BatchDataCPU GetBatchDataCPU(const BatchData & batch_data, bool copy_data, bool compacted)
{
    BatchDataCPU compress_data_cpu(batch_data.ptrs(), batch_data.sizes(), batch_data.data(), batch_data.size(), copy_data, compacted);
    return compress_data_cpu;
}

BatchDataCPU compress(const BatchDataCPU & input_data_cpu)
{
    //size_t cpu_cores = getNumberOfPhysicalCPUCores() * 2;
    size_t chunk_nums = input_data_cpu.size();

    //auto thread_num = std::min(cpu_cores, chunk_nums);

    BatchDataCPU compress_data_cpu(LZ4_compressBound(chunk_size), chunk_nums);

    //std::atomic<uint64_t> comp_bytes{0};

    //size_t chunks_per_thread = chunk_nums / thread_num;

    /// Implement multi-threads to compress
    auto f = [&](size_t i) {
        // could use LZ4_compress_default or LZ4_compress_fast instead
        //size_t start = chunks_per_thread * i;
        //size_t end = (i == thread_num - 1) ? chunk_nums : chunks_per_thread * (i + 1);
        //for (size_t j = start; j < end; ++j)
        //{
        const int size = LZ4_compress_default(
            static_cast<const char *>(input_data_cpu.ptrs()[i]),
            static_cast<char *>(compress_data_cpu.ptrs()[i]),
            input_data_cpu.sizes()[i],
            compress_data_cpu.sizes()[i]);
            //1);
        if (size == 0)
        {
            throw std::runtime_error("LZ4 CPU failed to compress chunk " + std::to_string(i) + ".");
            }

            // set the actual compressed size
            compress_data_cpu.sizes()[i] = size;
        //}
    };

    std::vector<std::thread> compress_threads;
    compress_threads.reserve(chunk_nums);


    for (size_t i = 0; i < chunk_nums; ++i)
    {
        compress_threads.emplace_back(f, i);
    }
    Stopwatch sw;
    for (auto & th : compress_threads)
    {
        th.join();
    }
    auto ms = sw.elapsedMilliseconds();


    //std::cout << "Compressed ratio: " << (double)input_data_cpu.data_size() / comp_bytes << std::endl;
    std::cout << "CPU compression time: " << ms << " ms" << std::endl;
    std::cout << "CPU compression throughput (GB/s): " << (double)input_data_cpu.data_size() / (1.0e6 * ms) << std::endl << std::endl;
    return compress_data_cpu;
}

/// input_data: GPU uncompressed data
BatchDataCPU decompress(const BatchDataCPU & compress_data_cpu, const BatchData & input_data)
{
    size_t cpu_cores = getNumberOfPhysicalCPUCores() * 2;
    size_t chunk_nums = input_data.size();

    auto thread_num = std::min(cpu_cores, chunk_nums);
    size_t chunks_per_thread = chunk_nums / thread_num;

    BatchDataCPU decompress_data_cpu = GetBatchDataCPU(input_data, false);
    /// Implement multi-thread to decompress
    auto f = [&](int i) {
        size_t start = chunks_per_thread * i;
        size_t end = (i == thread_num - 1) ? chunk_nums : chunks_per_thread * (i + 1);

        for (size_t j = start; j < end; ++j)
        {
            const int size = LZ4_decompress_safe(
                static_cast<const char *>(compress_data_cpu.ptrs()[j]),
                static_cast<char *>(decompress_data_cpu.ptrs()[j]),
                compress_data_cpu.sizes()[j],
                decompress_data_cpu.sizes()[j]);
            if (size <= 0)
            {
                throw std::runtime_error("LZ4 CPU failed to decompress chunk " + std::to_string(j) + ".");
            }
        }
    };

    std::vector<std::thread> decompress_threads;
    decompress_threads.reserve(thread_num);

    for (size_t i = 0; i < thread_num; ++i)
    {
        decompress_threads.emplace_back(f, i);
    }

    Stopwatch sw;
    for (auto & th : decompress_threads)
    {
        th.join();
    }

    auto ms = sw.elapsedMilliseconds();

    double decompression_throughput = ((double)compress_data_cpu.data_size() / ms) * 1e-6;
    std::cout << "CPU decompression time: " << ms << " ms" << std::endl;
    std::cout << "CPU decompression throughput (GB/s): " << decompression_throughput << std::endl << std::endl;
    return decompress_data_cpu;
}
/// input_data: GPU uncompressed data
BatchDataCPU decompress(const BatchDataCPU & compress_data_cpu, size_t uncompressed_data_size)
{
    size_t chunk_nums = compress_data_cpu.size();

    BatchDataCPU decompress_data_cpu(nullptr, uncompressed_data_size, chunk_size, false);

    /// Implement multi-thread to decompress
    auto f = [&](int i) {
        const int size = LZ4_decompress_safe(
            static_cast<const char *>(compress_data_cpu.ptrs()[i]),
            static_cast<char *>(decompress_data_cpu.ptrs()[i]),
            compress_data_cpu.sizes()[i],
            decompress_data_cpu.sizes()[i]);
        if (size <= 0)
        {
            throw std::runtime_error("LZ4 CPU failed to decompress chunk " + std::to_string(i) + ".");
        }
    };

    std::vector<std::thread> decompress_threads;
    decompress_threads.reserve(chunk_nums);

    for (size_t i = 0; i < chunk_nums; ++i)
    {
        decompress_threads.emplace_back(f, i);
    }

    Stopwatch sw;
    for (auto & th : decompress_threads)
    {
        th.join();
    }

    auto ms = sw.elapsedMilliseconds();

    double decompression_throughput = ((double)compress_data_cpu.data_size() / ms) * 1e-6;
    std::cout << "CPU decompression time: " << ms << " ms" << std::endl;
    std::cout << "CPU decompression throughput (GB/s): " << decompression_throughput << std::endl << std::endl;
    return decompress_data_cpu;
}
}

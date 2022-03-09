#include <iostream>
#include <vector>
#include <Stopwatch.h>
#include <define.h>
#include <CpuLZ4Compressor.h>
#include <GpuLZ4Compressor.h>

using T = int;
const size_t vec_size = 1 << 25;
const size_t data_size = vec_size * sizeof(T);

int main()
{
    std::vector<T> vec(vec_size);
    srand(7);
    for (size_t i = 0; i < vec_size; ++i)
    {
        vec[i] = rand() % 10000;
    }

    BatchDataCPU input_data_cpu((char *)vec.data(), data_size, chunk_size);

    /// cpu compress
    auto compress_data_cpu = CpuLZ4::compress(input_data_cpu);
	compress_data_cpu.compact();

    Stopwatch sw;
    BatchData compress_data(compress_data_cpu, true, true);

    auto ms = sw.elapsedMilliseconds();
    std::cout << "Elapsed time of copy data from CPU to GPU: " << ms << " ms" << std::endl;

    /// gpu decompress
    auto decompress_data = GpuLZ4::decompress(compress_data, input_data_cpu);

    /// gpu compress
    auto compress_data_new = GpuLZ4::compress(decompress_data);

    //auto m_sizes = std::vector<size_t>(compress_data_new.size());
    //CUDA_CHECK(cudaMemcpy(m_sizes.data(), compress_data_new.sizes(), compress_data_new.size() * sizeof(void *), cudaMemcpyDeviceToHost));
//
    //size_t co_data_size = std::accumulate(m_sizes.begin(), m_sizes.end(), static_cast<size_t>(0));
    //std::cout << "compress data size = " << compress_data_cpu.data_size() << ", gpu compress data size = " << co_data_size << std::endl;

    //compress_data_new.compact();
    std::cout << "compress_data_new.data_size() = " << compress_data_new.data_size() << std::endl;

    sw.restart();
    /// copy data
    auto compress_data_cpu_new = CpuLZ4::GetBatchDataCPU(compress_data_new, true);
    ms = sw.elapsedMilliseconds();
    std::cout << "Elapsed time of copy data from GPU to CPU: " << ms << " ms" << std::endl;
    /// cpu decompress
    auto decompress_data_cpu = CpuLZ4::decompress(compress_data_cpu_new, decompress_data);

    std::vector<T> res(vec_size);
    std::memcpy((char *)res.data(), decompress_data_cpu.data(), data_size);

    for (size_t i = 0; i < vec_size; ++i)
    {
        if (res[i] != vec[i])
        {
            std::cout << "Test Failed." << std::endl;
            return 1;
        }
    }
    std::cout << "Test Pass!!!" << std::endl;
}

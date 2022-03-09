#include <CpuLZ4Compressor.h>
#include <iostream>
#include <vector>
#include <GpuLZ4Compressor.h>

int main()
{
    std::vector<int> vec(1024, 0);
    for (size_t i = 0; i < 1024; ++i)
    {
        vec[i] = i;
    }

    auto max_compressed_size = CpuLZ4::getMaxCompressedDataSize(1024 * 4);

    char * buf = new char[max_compressed_size];

    auto compressed_size = CpuLZ4::compress((char *)vec.data(), 1024 * 4, buf);

    std::cout << "Finish compress. compressed_size = " << compressed_size << std::endl;

	uint8_t * gpu_src;

    CUDA_CHECK(cudaMalloc(&gpu_src, compressed_size));
    CUDA_CHECK(cudaMemcpy(gpu_src, buf, compressed_size, cudaMemcpyHostToDevice));

    std::cout << "Finish cudaMemcpy" << std::endl;

    uint8_t* reverse_dst;
    GpuLZ4::decompress(gpu_src, compressed_size, 1024 * 4, reverse_dst);

    std::cout << "Finish decompress" << std::endl;

    std::vector<int> res(1024, 0);
    CUDA_CHECK(
        cudaMemcpy(res.data(), reverse_dst, 1024 * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 1024; ++i) {
      if (res[i] != i) {
        std::cout << "Test Failed. res[i] = " << res[i] << ", i = " << i
                  << std::endl;
        delete[] buf;
        cudaFree(reverse_dst);
        return 1;
      }
    }
    std::cout << "Test Pass!!!" << std::endl;
    delete[] buf;
    cudaFree(reverse_dst);
    CUDA_CHECK(cudaFree(reverse_dst));
    return 0;
}

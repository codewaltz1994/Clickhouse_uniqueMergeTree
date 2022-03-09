#include <fstream>
#include <iostream>
#include <vector>
#include <nvcomp_common_deps/hlif_shared_types.hpp>
#include <CpuLZ4Compressor.h>
#include <GpuLZ4Compressor.h>

const size_t uncompressed_data_size = 1024 * 4;
const size_t tuple_size = 1024;

int main()
{
    std::vector<int> vec(tuple_size);
    for (size_t i = 0; i < tuple_size; ++i)
    {
        vec[i] = i;
    }
    uint8_t * src, * dst;
    CUDA_CHECK(cudaMalloc(&src, uncompressed_data_size));
    CUDA_CHECK(cudaMemcpy(src, vec.data(), uncompressed_data_size, cudaMemcpyHostToDevice));

    auto compressed_size = GpuLZ4::compress(src, uncompressed_data_size, dst);

    std::cout << "Finish compress, compressed_size = " << compressed_size << std::endl;
    uint8_t * buf = new uint8_t[compressed_size - 100];
    buf[0] = 243;
    buf[1] = 255;
    buf[2] = 255;
    buf[3] = 255;
    buf[4] = 241;
    CUDA_CHECK(cudaMemcpy(buf + 5, dst + 105, compressed_size - 105, cudaMemcpyDeviceToHost));

    std::vector<int> res(tuple_size);

    std::ofstream file("./gpu_compress.txt");
    for (size_t i = 0; i < compressed_size - 100; ++i)
    {
        int a = *(uint8_t *)(buf + i);
        file << a << "\n";
    }

    int return_decom_size = CpuLZ4::decompress((char *)(buf), compressed_size - 100, (char *)res.data(), uncompressed_data_size);

    std::cout << "Finish decompress." << std::endl;

    if (uncompressed_data_size != return_decom_size)
    {
        std::cout << "Test Failed."
                  << " return_decom_size = " << return_decom_size << std::endl;
        CUDA_CHECK(cudaFree(src));
        CUDA_CHECK(cudaFree(dst));
        delete[] buf;
        return 1;
    }

    for (size_t i = 0; i < tuple_size; ++i)
    {
        if (res[i] != i)
        {
            std::cout << "Test Failed. res[i] = " << res[i] << ", i = " << i << std::endl;
            CUDA_CHECK(cudaFree(src));
            CUDA_CHECK(cudaFree(dst));
            delete[] buf;
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    delete[] buf;
    std::cout << "Test Passed!!!" << std::endl;
    return 0;
}

#include <fstream>
#include <vector>
#include <GpuLZ4Compressor.h>

const size_t uncompressed_data_size = 2048 * 4;
const size_t tuple_size = 2048;

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

    uint8_t * buf = new uint8_t[8320];
    CUDA_CHECK(cudaMemcpy(buf, dst, 8320, cudaMemcpyDeviceToHost));

    std::ofstream file("./gpu_compress.txt");
    for (size_t i = 96; i < 8320; ++i)
    {
        int a = *(uint8_t *)(buf + i);
        file << a << " ";
    }

    uint8_t * reverse_dst;
    GpuLZ4::decompress(dst, compressed_size, uncompressed_data_size, reverse_dst);

    std::vector<int> res(tuple_size, 0);
    CUDA_CHECK(cudaMemcpy(res.data(), reverse_dst, uncompressed_data_size, cudaMemcpyDeviceToHost));
    std::cout << "Finish decompress" << std::endl; //" compressed_size = " << compressed_size << std::endl;

    for (int i = 0; i < tuple_size; ++i)
    {
        if (res[i] != i)
        {
            std::cout << "Test Failed.";
            cudaFree(src);
            cudaFree(dst);
            cudaFree(reverse_dst);
            return 1;
        }
    }
    std::cout << "Test Pass!!!" << std::endl;
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(reverse_dst));
    return 0;
}

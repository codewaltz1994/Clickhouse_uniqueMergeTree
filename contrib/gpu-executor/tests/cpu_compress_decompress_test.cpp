#include <fstream>
#include <iostream>
#include <vector>
#include <CpuLZ4Compressor.h>

const size_t uncompressed_data_size = 2048 * 4;
const size_t tuple_size = 2048;

int main()
{
    std::vector<int> vec(tuple_size, 0);
    for (size_t i = 0; i < tuple_size; ++i)
    {
        vec[i] = i;
    }

    auto max_compressed_size = CpuLZ4::getMaxCompressedDataSize(uncompressed_data_size);

    char * buf = new char[max_compressed_size];

    auto compressed_size = CpuLZ4::compress((char *)vec.data(), uncompressed_data_size, buf);

    std::ofstream file("./cpu_compress.txt");
    for (size_t i = 0; i < 8239; ++i)
    {
        int a = *(uint8_t *)(buf + i);
        file << a << "\n";
    }

    std::cout << "Finish compress. compressed_size = " << compressed_size << std::endl;

    std::vector<int> res(tuple_size, 0);

    std::cout << *(((int *)(buf)) + 2) << std::endl;
    auto return_decom_size = CpuLZ4::decompress(buf, compressed_size, (char *)res.data(), uncompressed_data_size);

    std::cout << "Finish decompress." << std::endl;

    if (uncompressed_data_size != return_decom_size)
    {
        std::cout << "Test Failed."
                  << " return_decom_size = " << return_decom_size << std::endl;
        delete[] buf;
        return 1;
    }

    for (size_t i = 0; i < tuple_size; ++i)
    {
        if (res[i] != i)
        {
            std::cout << "Test Failed."
                      << " res[i] = " << res[i] << ", i = " << i << std::endl;
            delete[] buf;
            return 1;
        }
    }

    delete[] buf;
    std::cout << "Test Passed!!!" << std::endl;
    return 0;
}

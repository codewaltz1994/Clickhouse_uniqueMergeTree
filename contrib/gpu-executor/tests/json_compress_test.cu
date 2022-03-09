#include <fstream>
#include <iostream>
#include <vector>
#include <Stopwatch.h>
#include <assert.h>
#include <CpuLZ4Compressor.h>

constexpr size_t uncompressed_size = 1024 * 1024 * 1024;

int main()
{
    std::ifstream is("/data/fenglv/final/gpu-executor/ol_cdump.json");
    std::vector<uint8_t> data(uncompressed_size);
    is.read((char *)data.data(), uncompressed_size);

    size_t max_comressed_size = CpuLZ4::getMaxCompressedDataSize(uncompressed_size);
    std::vector<uint8_t> compressed_data(max_comressed_size);

    Stopwatch sw;
    size_t compressed_size = CpuLZ4::compress((char *)data.data(), uncompressed_size, (char *)compressed_data.data());
    std::cout << "uncompressed_size = " << uncompressed_size / 1024.0 / 1024.0
              << " MB, compressed_size = " << compressed_size / 1024.0 / 1024.0 << " MB" << std::endl;
    auto seconds = sw.elapsedSeconds();
    std::cout << "time in compress " << seconds << " seconds" << std::endl;
    std::cout << "Compress ratio = " << (double)uncompressed_size / compressed_size << std::endl;

    std::vector<uint8_t> uncompressed_data(uncompressed_size);
    sw.restart();
    auto return_uncompressed_size
        = CpuLZ4::decompress((char *)compressed_data.data(), compressed_size, (char *)uncompressed_data.data(), uncompressed_size);
    seconds = sw.elapsedSeconds();
    std::cout << "time in decompress " << seconds << " seconds" << std::endl;
    assert(return_uncompressed_size == uncompressed_size);
}

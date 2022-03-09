#include <fstream>
#include <iostream>
#include <vector>
#include <BatchData.h>
#include <BatchDataCPU.h>
#include <Stopwatch.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <define.h>
//#include <CpuLZ4Compressor.h>
//#include <GpuLZ4Compressor.h>

constexpr size_t uncompressed_size = 1024 * 1024 * 1024;

void f1()
{
    std::ifstream is("/data/fenglv/final/gpu-executor/ol_cdump.json");
    std::vector<uint8_t> data(uncompressed_size);
    is.read((char *)data.data(), uncompressed_size);

    char * device_buf;
    Stopwatch sw;
    cudaMalloc(&(device_buf), uncompressed_size);
    cudaMemcpy(device_buf, data.data(), uncompressed_size, cudaMemcpyHostToDevice);
    auto ms = sw.elapsedMilliseconds();
    std::cout << "cudaMemcpy host to device time = " << ms << " ms" << std::endl;

    sw.restart();
    cudaMemcpy(data.data(), device_buf, uncompressed_size, cudaMemcpyDeviceToHost);
    ms = sw.elapsedMilliseconds();
    std::cout << "cudaMemcpy device to host time = " << ms << " ms" << std::endl;
    cudaFree(device_buf);
}

void f2()
{
}
int main()
{
    f1();
    f1();
    f2();
    f2();
    f2();
}

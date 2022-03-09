#include <vector>
#include <Stopwatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <util.h>

using T = int;
const size_t vec_size = 1 << 18;
const size_t data_size = vec_size * sizeof(T);

int main()
{
    std::vector<int> vec(vec_size);

    char * device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, data_size));

    Stopwatch sw;
    CUDA_CHECK(cudaMemcpy(device_ptr, vec.data(), data_size, cudaMemcpyHostToDevice));
    auto ms = sw.elapsedNanoseconds();
    std::cout << "Elapsed time of copy data from CPU to GPU: " << ms << " ns" << std::endl;

    sw.restart();
    CUDA_CHECK(cudaMemcpy(vec.data(), device_ptr, data_size, cudaMemcpyDeviceToHost));
    ms = sw.elapsedNanoseconds();
    std::cout << "Elapsed time of copy data from GPU to CPU: " << ms << " ns" << std::endl;
    CUDA_CHECK(cudaFree(device_ptr));
}

#include <iostream>
#include <Stopwatch.h>
#include <define.h>
//#include <CpuLZ4Compressor.h>
//#include <GpuLZ4Compressor.h>
#include "UserDefinedFunctions.h"

std::vector<std::complex<float>> Functions::fft(const float * data, size_t size)
{
    cufftComplex * complex_data = new cufftComplex[size];
    for (size_t i = 0; i < size; ++i)
    {
        complex_data[i].x = data[i];
        complex_data[i].y = 0;
    }

    cufftComplex * dfft_data;

    cudaMalloc((void **)&dfft_data, size * sizeof(cufftComplex));

    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        exit(-1);
    }

    Stopwatch sw;
    cudaMemcpy(dfft_data, complex_data, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to Memcpy\n");
        exit(-1);
    }
    auto ms = sw.elapsedMilliseconds();
    std::cout << "Memcpy time: " << ms << " ms, data size = " << size * sizeof(cufftComplex) << std::endl;

    cufftHandle plan;
    if (cufftPlan1d(&plan, size, CUFFT_C2C, 1) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "Cuda error: Failed to Handle\n");
        exit(-1);
    }

    sw.restart();
    if (cufftExecC2C(plan, static_cast<cufftComplex *>(dfft_data), static_cast<cufftComplex *>(dfft_data), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "Cuda error: Failed to exec\n");
        exit(-1);
    }
    ms = sw.elapsedMilliseconds();
    std::cout << "FFT execute time: " << ms << " ms" << std::endl;

    cudaDeviceSynchronize();

    sw.restart();
    cudaMemcpy(complex_data, dfft_data, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to Memcpy\n");
		exit(-1);
	}
    ms = sw.elapsedMilliseconds();
    std::cout << "Memcpy time: " << ms << " ms, data size = " << size * sizeof(cufftComplex) << std::endl << std::endl;

    std::vector<std::complex<float>> res;
    res.resize(size);

    for (size_t i = 0; i < size; ++i)
    {
        res[i] = {complex_data[i].x, complex_data[i].y};
    }
	delete []complex_data;
	cudaFree(dfft_data);
	return res;
}

#if 0
std::vector<std::complex<float>> Functions::fftWithCompress(const std::vector<float> & data)
{
    size_t size = data.size();
    cufftComplex * complex_data = new cufftComplex[size];
    for (size_t i = 0; i < size; ++i)
    {
        complex_data[i].x = data[i];
        complex_data[i].y = 0;
    }

    BatchDataCPU batch_data_cpu((char *)complex_data, size * sizeof(cufftComplex), chunk_size);

    /// compress
    auto compress_data_cpu = CpuLZ4::compress(batch_data_cpu);
    compress_data_cpu.compact();

    Stopwatch sw;

    BatchData compress_data(compress_data_cpu, true, true);
    auto ms = sw.elapsedMilliseconds();
    std::cout << "Memcpy time(with compress): " << ms << " ms, data size = " << compress_data_cpu.data_size()
              << ", compress ratio = " << batch_data_cpu.data_size() * 1.0 / compress_data_cpu.data_size() << std::endl;

    auto decompress_data = GpuLZ4::decompress(compress_data, batch_data_cpu);

    cufftHandle plan;
    if (cufftPlan1d(&plan, size, CUFFT_C2C, 1) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "Cuda error: Failed to Handle\n");
        exit(-1);
    }

    sw.restart();
    if (cufftExecC2C(
            plan,
            reinterpret_cast<cufftComplex *>(decompress_data.data()),
            reinterpret_cast<cufftComplex *>(decompress_data.data()),
            CUFFT_FORWARD)
        != CUFFT_SUCCESS)
    {
        fprintf(stderr, "Cuda error: Failed to exec\n");
        exit(-1);
    }
    ms = sw.elapsedMilliseconds();
    std::cout << "FFT execute time(with compress): " << ms << " ms" << std::endl;

    cudaDeviceSynchronize();

#    if 0
	sw.restart();
    cudaMemcpy(complex_data, decompress_data.data(), size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to Memcpy\n");
		exit(-1);
	}
	ms = sw.elapsedMilliseconds();
    std::cout << "Memcpy time(with non-compress): " << ms << " ms, data size = " << size * sizeof(cufftComplex) << std::endl << std::endl;
#    endif
	auto compress_data_new = GpuLZ4::compress(decompress_data);
	
	sw.restart();
	auto compress_data_cpu_new = CpuLZ4::GetBatchDataCPU(compress_data_new, true);
	ms = sw.elapsedMilliseconds();
    std::cout << "Memcpy time(with compress): " << ms << " ms, data size = " << compress_data_cpu_new.data_size() << std::endl;

    auto decompress_data_cpu = CpuLZ4::decompress(compress_data_cpu_new, decompress_data);
	
	auto * complex_data_new = reinterpret_cast<cufftComplex *>(decompress_data_cpu.data());

    std::vector<std::complex<float>> res;
    res.resize(size);

    for (size_t i = 0; i < size; ++i)
    {
		res[i] = {complex_data_new[i].x, complex_data_new[i].y};
        //res[i] = {complex_data[i].x, complex_data[i].y};
    }

    return res;
}
#endif

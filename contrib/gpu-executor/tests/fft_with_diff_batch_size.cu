#include <iostream>
#include <vector>
#include <Stopwatch.h>
#include <UserDefinedFunctions.h>
#include <util.h>

constexpr size_t max_batch_size = 1 << 26;

double func(size_t batch_size)
{
    Stopwatch sw;
    auto vec = genRandomWithSizeAndMaxValue<float>(batch_size);
    Functions::fft(vec.data(), vec.size());
    return sw.elapsedMilliseconds();
}

int main()
{
    std::vector<double> times;
    for (size_t batch_size = 1; batch_size <= max_batch_size; batch_size <<= 1)
    {
        times.emplace_back(func(batch_size));
    }
    size_t batch_size = 1;
    for (size_t i = 1; i <= 27; ++i)
    {
        std::cout << "batch_size = " << batch_size << " , lantency = " << times[i - 1] << std::endl;
        batch_size <<= 1;
    }
}

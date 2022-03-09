#include <Stopwatch.h>
#include <UserDefinedFunctions.h>
#include <util.h>

const uint64_t batch_size = 1 << 6;

double func1(const std::vector<float> & vec)
{
	Stopwatch sw;
    auto res = Functions::fft((float *)vec.data(), vec.size());
    for (size_t i = 0; i < res.size(); ++i)
    {
        std::cout << res[i] << std::endl;
    }
    auto ms = sw.elapsedMilliseconds();
    return ms;
}

double func2(const std::vector<float> & vec)
{
	Stopwatch sw;
    Functions::fftWithCompress(vec);
    auto ms = sw.elapsedMilliseconds();
    return ms;
}
void just_run_f1(const std::vector<float> & vec)
{
    func1(vec);
    std::vector<double> vec1;
    for (size_t i = 0; i < 3; ++i)
    {
        vec1.emplace_back(func1(vec));
    }
    for (auto t : vec1)
    {
        std::cout << t << "\t";
    }
    std::cout << std::endl;
    std::cout << "average run time without compress = " << std::accumulate(vec1.begin(), vec1.end(), 0.0) / 3.0 << std::endl;
}

void just_run_f2(const std::vector<float> & vec)
{
	func2(vec);
    std::vector<double> vec2;
    for (size_t i = 0; i < 3; ++i)
    {
        vec2.emplace_back(func2(vec));
    }
    for (auto t : vec2)
    {
        std::cout << t << "\t";
    }
    std::cout << std::endl;
    std::cout << "average run time with compress = " << std::accumulate(vec2.begin(), vec2.end(), 0.0) / 3.0 << std::endl;
}

int main(int argc, char ** argv)
{
    auto vec = genRandomWithSizeAndMaxValue<float>(batch_size, 255);
    for (auto & i : vec)
    {
        i /= 255;
    }
    just_run_f1(vec);
    //if (argc == 1)
		//just_run_f1(vec);
    //else if (argc == 2)
        //just_run_f2(vec);
    //else
    //{
		//just_run_f1(vec);
        //just_run_f2(vec);
    //}
    return 0;
}

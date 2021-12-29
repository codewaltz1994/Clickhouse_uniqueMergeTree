#include <Columns/ColumnTuple.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>

#include <cmath>
#include <complex>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_COLUMN;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

struct FFTImpl
{
    static constexpr float PI = 3.1415926536;

    static void
    fft(const PaddedPODArray<Float32> & input_real,
        const PaddedPODArray<Float32> & input_image,
        PaddedPODArray<Float32> & output_real,
        PaddedPODArray<Float32> & output_image)
    {
        auto data_size = input_real.size();

        output_real.resize(data_size);
        output_image.resize(data_size);

        std::vector<std::complex<float>> a;
        a.resize(data_size);
        for (size_t i = 0; i < data_size; ++i)
        {
            a[i] = {input_real[i], input_image[i]};
        }
        fftImpl(a);

        for (size_t i = 0; i < data_size; ++i)
        {
            output_real[i] = a[i].real();
            output_image[i] = a[i].imag();
        }
    }

private:
    static void fftImpl(std::vector<std::complex<float>> & a)
    {
        int n = a.size();
        if (n <= 1)
            return;
        std::vector<std::complex<float>> a0(n / 2), a1(n / 2);
        for (int i = 0; 2 * i < n - 1; i++)
        {
            a0[i] = a[2 * i];
            a1[i] = a[2 * i + 1];
        }
        fftImpl(a0);
        fftImpl(a1);

        double ang = 2 * PI / n;
        std::complex<float> w(1), wn(cos(ang), sin(ang));
        for (int i = 0; 2 * i < n - 1; i++)
        {
            a[i] = a0[i] + w * a1[i];
            a[i + n / 2] = a0[i] - w * a1[i];
            w *= wn;
        }
    }
};

class FunctionFFTCpu : public IFunction
{
public:
    static constexpr auto name = "fftCpu";

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionFFTCpu>(); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isFloat(arguments[0]) && !isFloat(arguments[1]))
            throw Exception(
                "Illegal type " + arguments[0]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        return std::make_shared<DataTypeTuple>(arguments);
    }

    bool useDefaultImplementationForConstants() const override { return true; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t) const override
    {
        const auto & real_col = arguments[0].column;
        const auto & image_col = arguments[1].column;
        if (const ColumnVector<Float32> * col1 = checkAndGetColumn<ColumnVector<Float32>>(real_col.get()))
        {
            if (const ColumnVector<Float32> * col2 = checkAndGetColumn<ColumnVector<Float32>>(image_col.get()))
            {
                auto col_res1 = ColumnVector<Float32>::create();
                auto col_res2 = ColumnVector<Float32>::create();
                auto & data_col1 = col_res1->getData();
                auto & data_col2 = col_res2->getData();
                FFTImpl::fft(col1->getData(), col2->getData(), data_col1, data_col2);
                MutableColumns columns(2);
                columns[0] = std::move(col_res1);
                columns[1] = std::move(col_res2);
                return ColumnTuple::create(std::move(columns));
            }
            throw Exception(
                "Illegal column " + arguments[0].column->getName() + " of argument 1 of function " + getName(), ErrorCodes::ILLEGAL_COLUMN);
        }
        throw Exception(
            "Illegal column " + arguments[1].column->getName() + " of argument 0 of function " + getName(), ErrorCodes::ILLEGAL_COLUMN);
    }
};

}

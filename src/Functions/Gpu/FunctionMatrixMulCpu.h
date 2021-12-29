#include <Columns/ColumnConst.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_COLUMN;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

struct MatrixMulImpl
{
    static void matrixMul(
        const PaddedPODArray<Float32> & a,
        const PaddedPODArray<Float32> & b,
        PaddedPODArray<Float32> & c,
        size_t batch_size,
        size_t matrix_len)
    {
        if (a.empty())
            return;
        auto result_size = a.size();
        c.resize(result_size);
        for (size_t batch = 0; batch < batch_size; ++batch)
        {
            size_t offset = batch * matrix_len * matrix_len;
            for (size_t i = 0; i < matrix_len; ++i)
            {
                for (size_t j = 0; j < matrix_len; ++j)
                {
                    for (size_t k = 0; k < matrix_len; ++k)
                    {
                        c[offset + i * matrix_len + j] += a[offset + i * matrix_len + k] * b[offset + k * matrix_len + j];
                    }
                }
            }
        }
    }
};

class FunctionMatrixMulCpu : public IFunction
{
public:
    static constexpr auto name = "mmCpu";

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionMatrixMulCpu>(); }

    String getName() const override { return name; }

    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {2, 3}; }
    size_t getNumberOfArguments() const override { return 4; }

    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isFloat(arguments[0]))
            throw Exception(
                "Illegal type " + arguments[0]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        if (!isFloat(arguments[1]))
            throw Exception(
                "Illegal type " + arguments[1]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        if (!isUnsignedInteger(arguments[2]))
            throw Exception(
                "Illegal type " + arguments[2]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        if (!isUnsignedInteger(arguments[3]))
            throw Exception(
                "Illegal type " + arguments[3]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        return arguments[0];
    }

    bool useDefaultImplementationForConstants() const override { return true; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t) const override
    {
        const auto * col0 = arguments[0].column.get();
        const auto * col1 = arguments[1].column.get();
        const auto * col2 = arguments[2].column.get();
        const auto * col3 = arguments[3].column.get();
        if (const auto * col0_vec = checkAndGetColumn<ColumnVector<Float32>>(col0))
        {
            if (const auto * col1_vec = checkAndGetColumn<ColumnVector<Float32>>(col1))
            {
                if (const auto * col2_const = checkAndGetColumn<ColumnConst>(col2))
                {
                    if (const auto * col3_const = checkAndGetColumn<ColumnConst>(col3))
                    {
                        UInt64 batch_size = col2_const->getValue<UInt64>();
                        UInt64 matrix_len = col3_const->getValue<UInt64>();
                        if (!col0_vec->empty() && batch_size * matrix_len * matrix_len != col0_vec->size())
                        {
                            throw Exception(
                                "Illegal columns of argument of function " + getName()
                                    + ", batch_size plus matrix_size should equals to data size, batch_size = " + toString(batch_size)
                                    + ", matrix_len = " + toString(matrix_len) + ", vec_size = " + toString(col0_vec->size()),
                                ErrorCodes::ILLEGAL_COLUMN);
                        }
                        auto res = ColumnVector<Float32>::create();
                        MatrixMulImpl::matrixMul(col0_vec->getData(), col1_vec->getData(), res->getData(), batch_size, matrix_len);
                        return res;
                    }
                    throw Exception(
                        "Illegal column " + arguments[3].column->getName() + " of argument 3 of function " + getName(),
                        ErrorCodes::ILLEGAL_COLUMN);
                }
                throw Exception(
                    "Illegal column " + arguments[2].column->getName() + " of argument 2 of function " + getName(),
                    ErrorCodes::ILLEGAL_COLUMN);
            }
            throw Exception(
                "Illegal column " + arguments[1].column->getName() + " of argument 1 of function " + getName(), ErrorCodes::ILLEGAL_COLUMN);
        }
        throw Exception(
            "Illegal column " + arguments[0].column->getName() + " of argument 0 of function " + getName(), ErrorCodes::ILLEGAL_COLUMN);
    }
};

}

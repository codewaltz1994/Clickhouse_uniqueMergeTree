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


class FunctionMatrixMul : public IFunction
{
public:
    static constexpr auto name = "mm";

    FunctionMatrixMul() { is_gpu_function = true; }

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionMatrixMul>(); }

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
                "Illegal type " + arguments[0]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        if (!isUnsignedInteger(arguments[2]))
            throw Exception(
                "Illegal type " + arguments[1]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        if (!isUnsignedInteger(arguments[3]))
            throw Exception(
                "Illegal type " + arguments[2]->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        return arguments[1];
    }

    bool useDefaultImplementationForConstants() const override { return true; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName &, const DataTypePtr &, size_t input_rows_count) const override
    {
        return DataTypeFloat32{}.createColumnConst(input_rows_count, 0);
    }
};

}

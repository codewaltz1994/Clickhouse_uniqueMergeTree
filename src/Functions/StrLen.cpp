#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/IFunction.h>
#include <Functions/IFunctionImpl.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_COLUMN;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int TOO_LARGE_STRING_SIZE;
}

namespace
{
    class FunctionStrLen : public IFunction
    {

    public:
        static constexpr auto name = "strLen";
        static FunctionPtr create(const Context &) { return std::make_shared<FunctionStrLen>(); }

        String getName() const override { return name; }

        size_t getNumberOfArguments() const override { return 1; }

        DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
        {
            if (!isString(arguments[0]))
                throw Exception(
                    "Illegal type " + arguments[0]->getName() + " of argument of function " + getName(),
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
            return std::make_shared<DataTypeUInt64>();
        }

        bool useDefaultImplementationForConstants() const override { return true; }

        ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t) const override
        {
            const auto & strcolumn = arguments[0].column;

            if (const ColumnString * col = checkAndGetColumn<ColumnString>(strcolumn.get()))
            {
                const auto & offsets = col->getOffsets();

                auto col_res = ColumnVector<UInt64>::create();
                auto & res_data = col_res->getData();
                res_data.resize(offsets.size());

                for (size_t i = 0; i < offsets.size(); ++i)
                    res_data[i] = offsets[i] - offsets[i - 1] - 1;

                return col_res;
            }
            throw Exception(
                "Illegal column " + arguments[0].column->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_COLUMN);
        }
    };

}

void registerFunctionStrLen(FunctionFactory & factory)
{
    factory.registerFunction<FunctionStrLen>();
}
}

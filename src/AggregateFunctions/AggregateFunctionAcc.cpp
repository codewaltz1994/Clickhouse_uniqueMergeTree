#include <AggregateFunctions/AggregateFunctionAcc.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <AggregateFunctions/Helpers.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

struct Settings;

namespace
{
    template <template <typename> class Data>
    AggregateFunctionPtr createAggregateFunctionAcc(const std::string & name, const DataTypes & arguments, const Array &, const Settings *)
    {
        if (arguments.size() != 1)
            throw Exception(
                "Aggregate function " + name + " requires two timestamps argument.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        const auto * arg = arguments[0].get();

        if (!isNativeNumber(arg) && !isDate(arg) && !isDateTime(arg))
            throw Exception(
                "Illegal type " + arg->getName() + " of argument of aggregate function " + name
                    + ", must be native integral type, Date/DateTime or Float",
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        AggregateFunctionPtr res(createWithBasicNumberOrDateOrDateTime<AggregateFunctionAcc, Data>(*arguments[0], arguments));

        if (res)
            return res;

        throw Exception(
            "Illegal type " + arguments.front().get()->getName() + " of argument of aggregate function " + name
            + ", must be native integral type, Date/DateTime or Float",
            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
}

}

void registerAggregateFunctionAcc(AggregateFunctionFactory & factory)
{
    factory.registerFunction("acc", createAggregateFunctionAcc<AggregateFunctionAccData>);
}

}

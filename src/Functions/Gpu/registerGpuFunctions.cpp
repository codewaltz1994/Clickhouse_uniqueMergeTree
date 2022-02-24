#include <Functions/FunctionFactory.h>
#include <Functions/Gpu/FunctionMatrixMul.h>

namespace DB
{
void registerFunctionsGUDF(FunctionFactory & factory)
{
    factory.registerFunction<FunctionMatrixMul>();
}

}

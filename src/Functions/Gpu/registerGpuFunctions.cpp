#include <Functions/FunctionFactory.h>
#include <Functions/Gpu/FunctionFFT.h>
#include <Functions/Gpu/FunctionFFTCpu.h>
#include <Functions/Gpu/FunctionMatrixMul.h>
#include <Functions/Gpu/FunctionMatrixMulCpu.h>

#include <Functions/Gpu/FunctionDBF.h>
#include <Functions/Gpu/FunctionEcho.h>
#include <Functions/Gpu/FunctionMTD.h>
#include <Functions/Gpu/FunctionPC.h>

namespace DB
{
void registerFunctionsGUDF(FunctionFactory & factory)
{
    factory.registerFunction<FunctionFFT>();
    factory.registerFunction<FunctionFFTCpu>();
    factory.registerFunction<FunctionMatrixMul>();
    factory.registerFunction<FunctionMatrixMulCpu>();

    factory.registerFunction<FunctionEcho>();
    factory.registerFunction<FunctionDBF>();
    factory.registerFunction<FunctionPC>();
    factory.registerFunction<FunctionMTD>();

    factory.registerAlias("sf1", FunctionEcho::name, FunctionFactory::CaseSensitive);
	factory.registerAlias("sf2", FunctionDBF::name, FunctionFactory::CaseSensitive);
	factory.registerAlias("sf3", FunctionPC::name, FunctionFactory::CaseSensitive);
    factory.registerAlias("sf4", FunctionMTD::name, FunctionFactory::CaseSensitive);
}

}

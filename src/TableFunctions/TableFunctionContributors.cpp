#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Storages/System/StorageSystemContributors.h>
#include <TableFunctions/ITableFunction.h>
#include <TableFunctions/TableFunctionContributors.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/parseColumnsListForTableFunction.h>
#include "registerTableFunctions.h"


namespace DB
{
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

ColumnsDescription TableFunctionContributors::getActualTableStructure(const Context & /*context*/) const
{
    return ColumnsDescription{{{"name", std::make_shared<DataTypeString>()}}};
}

StoragePtr TableFunctionContributors::executeImpl(
    const ASTPtr & /*ast_function*/, const Context & , const std::string & table_name, ColumnsDescription /*cached_columns*/) const
{
    auto res = StorageSystemContributors::create(StorageID(getDatabaseName(), table_name));
    res->startup();
    return res;
}

void registerTableFunctionContributors(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionContributors>();
}
}

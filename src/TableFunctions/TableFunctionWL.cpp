#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTFunction.h>
#include <Parsers/ASTLiteral.h>
#include <Storages/StorageBinary.h>
#include <Storages/StorageNull.h>
#include <TableFunctions/ITableFunction.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/TableFunctionWL.h>
#include <TableFunctions/parseColumnsListForTableFunction.h>
#include "registerTableFunctions.h"


namespace DB
{
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

void TableFunctionWL::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    const auto * function = ast_function->as<ASTFunction>();
    if (!function || !function->arguments || function->arguments->children.size() != 2)
        throw Exception("Table function '" + getName() + "' requires 2 arguments.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    const auto & arguments = function->arguments->children;

    path = evaluateConstantExpressionOrIdentifierAsLiteral(arguments[0], context)->as<ASTLiteral &>().value.get<String>();
    number = evaluateConstantExpressionOrIdentifierAsLiteral(arguments[1], context)->as<ASTLiteral &>().value.get<UInt64>();
}

ColumnsDescription TableFunctionWL::getActualTableStructure(ContextPtr) const
{
    return ColumnsDescription({{"binary", std::make_shared<DataTypeString>()}, {"frame", std::make_shared<DataTypeUInt32>()}});
}

StoragePtr TableFunctionWL::executeImpl(
    const ASTPtr & /*ast_function*/, ContextPtr, const std::string & table_name, ColumnsDescription /*cached_columns*/) const
{
    // auto columns = getActualTableStructure(context);
    auto res = StorageBinary::create(StorageID(getDatabaseName(), table_name), path, number);
    res->startup();
    return res;
}

void registerTableFunctionWL(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionWL>();
}
}

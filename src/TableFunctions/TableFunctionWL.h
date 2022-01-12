#pragma once

#include <TableFunctions/ITableFunction.h>
#include <Core/Types.h>


namespace DB
{

class TableFunctionWL : public ITableFunction
{
public:
    static constexpr auto name = "WL";
    std::string getName() const override { return name; }
private:
    StoragePtr executeImpl(const ASTPtr & ast_function, ContextPtr context, const String & table_name, ColumnsDescription cached_columns) const override;
    const char * getStorageTypeName() const override { return "WL"; }

    void parseArguments(const ASTPtr & ast_function, ContextPtr context) override;
    ColumnsDescription getActualTableStructure(ContextPtr context) const override;

    std::string path;
    UInt32 number;
};

}

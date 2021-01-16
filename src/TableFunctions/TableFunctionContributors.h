#pragma once

#include <TableFunctions/ITableFunction.h>
#include <Core/Types.h>


namespace DB
{
class TableFunctionContributors : public ITableFunction
{
public:
    static constexpr auto name = "contributors";
    std::string getName() const override { return name; }
private:
    StoragePtr executeImpl(const ASTPtr & ast_function, const Context & context, const String & table_name, ColumnsDescription cached_columns) const override;
    const char * getStorageTypeName() const override { return "Contributors"; }

    ColumnsDescription getActualTableStructure(const Context & context) const override;

};

}

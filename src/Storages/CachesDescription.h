#pragma once

#include <common/types.h>

#include <memory>
#include <vector>
#include <Core/Field.h>
#include <Parsers/ASTCacheDeclaration.h>
#include <Parsers/IAST_fwd.h>

namespace DB
{
template <bool is_table_cache_>
struct CacheDescription
{
    /// Definition AST of cache
    ASTPtr definition_ast = nullptr;

    bool is_table_cache = is_table_cache_;

    String column_name;
    String cluster_name;

    ASTCacheDeclaration::CachePolicy cache_policy = ASTCacheDeclaration::CachePolicy::NO;

    ASTCacheDeclaration::CacheTime start;
    ASTCacheDeclaration::CacheTime end;

    static CacheDescription getCacheFromAST(const ASTPtr & definition_ast)
    {
        CacheDescription<is_table_cache_> res;
        const auto & ast_cache = definition_ast->as<ASTCacheDeclaration &>();
        res.column_name = ast_cache.column_name;
        res.cluster_name = ast_cache.cluster_name;
        res.cache_policy = ast_cache.cache_policy;
        res.start = ast_cache.start;
        res.end = ast_cache.end;
        res.definition_ast = std::move(definition_ast);
        return res;
    }

    CacheDescription() = default;
};

class ColumnsCacheDescription : public std::vector<CacheDescription<false>>
{
public:
    auto findWithColAndCluster(const String & col_name, const String & cluster)
    {
        for (auto it = begin(); it != end(); ++it)
        {
            if (it->column_name == col_name && it->cluster_name == cluster)
                return it;
        }
        return end();
    }
};

class TableCacheDescription : public std::vector<CacheDescription<true>>
{
public:
    auto findWithCluster(const String & cluster)
    {
        for (auto it = begin(); it != end(); ++it)
        {
            if (it->cluster_name == cluster)
                return it;
        }
        return end();
    }
};
}

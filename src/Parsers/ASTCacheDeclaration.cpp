#include <Parsers/ASTCacheDeclaration.h>

#include <IO/Operators.h>
#include <Parsers/ASTFunction.h>
#include <Common/quoteString.h>


namespace DB
{
ASTPtr ASTCacheDeclaration::clone() const
{
    auto res = std::make_shared<ASTCacheDeclaration>();

    res->is_column_cache = is_column_cache;
    res->column_name = column_name;
    res->cluster_name = cluster_name;

    res->start = start;
    res->end = end;

    res->cache_policy = cache_policy;
    return res;
}


void ASTCacheDeclaration::formatImpl(const FormatSettings & s, FormatState &, FormatStateStacked) const
{
    if (is_column_cache)
        s.ostr << column_name << " ";
    s.ostr << (cache_policy == CachePolicy::FIFO ? "FIFO" : "CS");
    s.ostr << " ";
    s.ostr << cluster_name;
    s.ostr << "(INTERVAL ";
    s.ostr << start.value << " " << start.kind.toString();
    s.ostr << ", INTERVAL ";
    s.ostr << end.value << " " << end.kind.toString();
    s.ostr << ")";
}

}


#pragma once

#include <Parsers/IAST.h>
#include <Common/IntervalKind.h>

namespace DB
{
class ASTCacheDeclaration : public IAST
{
public:
    enum class CachePolicy
    {
        FIFO,
        CS,
        NO
    };

    struct CacheTime
    {
        IntervalKind kind;
        size_t value;
    };

    bool is_column_cache = false;

    String column_name;
    String cluster_name;

    CacheTime start;
    CacheTime end;

    CachePolicy cache_policy = CachePolicy::NO;

    /** Get the text that identifies this element. */
    String getID(char) const override { return "Cache"; }

    ASTPtr clone() const override;
    void formatImpl(const FormatSettings & s, FormatState & state, FormatStateStacked frame) const override;
};

}

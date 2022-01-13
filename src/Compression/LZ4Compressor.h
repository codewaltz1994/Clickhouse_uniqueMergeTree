#pragma once

#include <Compression/LZ4_decompress_faster.h>
#include <Core/Types.h>
#include <lz4.h>
#include <Common/Exception.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int CANNOT_DECOMPRESS;
}

/// Do we need a class?
UInt32 getMaxCompressedDataSize(int uncompressed_size)
{
    return LZ4_COMPRESSBOUND(uncompressed_size);
}

UInt32 LZ4Compress(const char * src, UInt32 src_size, char * dst)
{
    return LZ4_compress_default(src, dst, src_size, getMaxCompressedDataSize(src_size));
}

void LZ4Decompress(const char * src, UInt32 src_size, char * dst, UInt32 uncompressed_size)
{
    LZ4::PerformanceStatistics stat;
    bool success = LZ4::decompress(src, dst, src_size, uncompressed_size, stat);
    if (!success)
        throw Exception("Cannot decompress", ErrorCodes::CANNOT_DECOMPRESS);
}
}

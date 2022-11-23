#pragma once

#include <Storages/MergeTree/MergeTreePartition.h>
#include <Storages/UniqueMergeTree/PrimaryIndex.h>
#include <Common/LRUResourceCache.h>

namespace DB
{
class StorageUniqueMergeTree;

class PrimaryIndexCache : public LRUResourceCache<String, PrimaryIndex>
{
public:
    PrimaryIndexCache(StorageUniqueMergeTree & storage_, size_t max_cache_size) : LRUResourceCache(max_cache_size), storage(storage_) { }
    PrimaryIndexPtr getOrCreate(const String & partition_id, const MergeTreePartition & partition);

private:
    StorageUniqueMergeTree & storage;
};
}

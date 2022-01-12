#pragma once

#include <Storages/IStorage.h>
#include <base/logger_useful.h>

#include <atomic>
#include <shared_mutex>
#include <base/shared_ptr_helper.h>


namespace DB
{
class StorageBinary final : public shared_ptr_helper<StorageBinary>, public IStorage
{
    friend struct shared_ptr_helper<StorageBinary>;

public:
    std::string getName() const override { return "Binary"; }

    Pipe read(
        const Names & column_names,
        const StorageMetadataPtr & /*metadata_snapshot*/,
        SelectQueryInfo &,
        ContextPtr context,
        QueryProcessingStage::Enum,
        size_t max_block_size,
        unsigned) override;

    const std::string & path() const { return file_path; }

protected:
    friend class StorageBinarySource;

    /// From user's file
    explicit StorageBinary(const StorageID & storage_id_, const std::string & user_files_path, UInt64 number_);

private:
    std::string file_path;
    UInt64 number;
};

}

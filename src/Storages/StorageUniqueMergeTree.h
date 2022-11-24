#pragma once

#include <Core/Names.h>
#include <Storages/AlterCommands.h>
#include <Storages/IStorage.h>
#include <Storages/MergeTree/FutureMergedMutatedPart.h>
#include <Storages/MergeTree/MergePlainMergeTreeTask.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/MergeTreeDataMergerMutator.h>
#include <Storages/MergeTree/MergeTreeDataSelectExecutor.h>
#include <Storages/MergeTree/MergeTreeDataWriter.h>
#include <Storages/MergeTree/MergeTreeDeduplicationLog.h>
#include <Storages/MergeTree/MergeTreeMutationEntry.h>
#include <Storages/MergeTree/MergeTreeMutationStatus.h>
#include <Storages/MergeTree/MergeTreePartsMover.h>
#include <Storages/UniqueMergeTree/DeleteBitmapCache.h>
#include <Storages/UniqueMergeTree/PrimaryIndexCache.h>
#include <Storages/UniqueMergeTree/TableVersion.h>
#include <Common/MultiVersion.h>

#include <Disks/StoragePolicy.h>
#include <Common/SimpleIncrement.h>


namespace DB
{
class MergePlainUniqueMergeTreeTask;

/** See the description of the data structure in MergeTreeData.
  */
class StorageUniqueMergeTree final : public MergeTreeData
{
    friend class MergeTreeData;

public:
    static constexpr auto DELETE_DIR_NAME = "/deletes/";
    static constexpr auto TABLE_VERSION_NAME = "/table_version.dat";

    void startup() override;
    void flush() override;
    void shutdown() override;
    ~StorageUniqueMergeTree() override;

    std::string getName() const override { return "UniqueMergeTree"; }

    bool supportsParallelInsert() const override { return true; }

    bool supportsIndexForIn() const override { return true; }

    void read(
        QueryPlan & query_plan,
        const Names & column_names,
        const StorageSnapshotPtr & storage_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        unsigned num_streams) override;

    std::optional<UInt64> totalRows(const Settings &) const override;

    std::optional<UInt64> totalRowsByPartitionPredicate(const SelectQueryInfo &, ContextPtr) const override;
    std::optional<UInt64> totalBytes(const Settings &) const override;

    SinkToStoragePtr write(const ASTPtr & query, const StorageMetadataPtr & /*metadata_snapshot*/, ContextPtr context) override;

    /** Perform the next step in combining the parts.
      */
	bool optimize(
		const ASTPtr & query,
		const StorageMetadataPtr & /*metadata_snapshot*/,
		const ASTPtr & partition,
		bool final,
		bool deduplicate,
		const Names & deduplicate_by_columns,
		ContextPtr context) override;

    void drop() override;
    void truncate(const ASTPtr &, const StorageMetadataPtr &, ContextPtr, TableExclusiveLockHolder &) override;

    void checkTableCanBeDropped() const override;

    ActionLock getActionLock(StorageActionBlockType action_type) override;

    void onActionLockRemove(StorageActionBlockType action_type) override;

    CheckResults checkData(const ASTPtr & query, ContextPtr context) override;

    bool scheduleDataProcessingJob(BackgroundJobsAssignee & assignee) override;

    MergeTreeDeduplicationLog * getDeduplicationLog() { return deduplication_log.get(); }

    std::vector<MergeTreeMutationStatus> getMutationsStatus() const override { return {}; }

    auto & primaryIndexCache() { return primary_index_cache; }
    auto & deleteBitmapCache() { return delete_bitmap_cache; }

    auto currentVersion() const { return table_version.get(); }

    DataPartPtr findPartByInfo(const MergeTreePartInfo & part_info) const;

    MergeTreePartInfo findPartInfoByMinBlock(Int64 min_block) const;

    Block getSampleBlockWithDeleteOp() const;

    void fillNewPartName(MutableDataPartPtr & part, DataPartsLock & lock);

    ASTPtr getFetchIndexQuery(
        const MergeTreePartition & partition, const std::vector<Field> & min_key_values, const std::vector<Field> & max_key_values);

private:

    MergeTreeDataSelectExecutor reader;
    MergeTreeDataWriter writer;
    MergeTreeDataMergerMutator merger_mutator;

    std::unique_ptr<MergeTreeDeduplicationLog> deduplication_log;

    /// For block numbers.
    SimpleIncrement increment;

    /// Current table version
    MultiVersion<TableVersion> table_version;
    DeleteBitmapCache delete_bitmap_cache;
    PrimaryIndexCache primary_index_cache;
    std::mutex write_merge_lock;

    /// For clearOldParts
    AtomicStopwatch time_after_previous_cleanup_parts;
    /// For clearOldTemporaryDirectories.
    AtomicStopwatch time_after_previous_cleanup_temporary_directories;

    std::atomic<bool> shutdown_called {false};
    std::atomic<bool> flush_called {false};

    bool updatePrimaryIndexAndDeletes(
        const MergeTreePartition & partition,
        const ColumnPtr & delete_key_column,
        const std::vector<Field> & delete_min_values,
        const std::vector<Field> & delete_max_values,
        ContextPtr local_context);

    void loadTableVersion(bool attach);

    /// Load and initialize deduplication logs. Even if deduplication setting
    /// equals zero creates object with deduplication window equals zero.
    void loadDeduplicationLog();

    /** Determines what parts should be merged and merges it.
      * If aggressive - when selects parts don't takes into account their ratio size and novelty (used for OPTIMIZE query).
      * Returns true if merge is finished successfully.
      */
    bool merge(
        bool aggressive,
        const String & partition_id,
        bool final,
        bool deduplicate,
        const Names & deduplicate_by_columns,
        const MergeTreeTransactionPtr & txn,
        String * out_disable_reason = nullptr,
        bool optimize_skip_merged_partitions = false);

    /// Make part state outdated and queue it to remove without timeout
    /// If force, then stop merges and block them until part state became outdated. Throw exception if part doesn't exists
    /// If not force, then take merges selector and check that part is not participating in background operations.
    MergeTreeDataPartPtr outdatePart(MergeTreeTransaction * txn, const String & part_name, bool force);
    ActionLock stopMergesAndWait();

    friend struct CurrentlyMergingPartsTagger;

    struct PartVersionWithName
    {
        Int64 version;
        String name;

        bool operator <(const PartVersionWithName & s) const
        {
            return version < s.version;
        }
    };

    std::shared_ptr<MergeMutateSelectedEntry> selectPartsToMerge(
        const StorageMetadataPtr & metadata_snapshot,
        bool aggressive,
        const String & partition_id,
        bool final,
        String * disable_reason,
        TableLockHolder & table_lock_holder,
        std::unique_lock<std::mutex> & lock,
        const MergeTreeTransactionPtr & txn,
        bool optimize_skip_merged_partitions = false,
        SelectPartsDecision * select_decision_out = nullptr);

    /// Returns maximum version of a part, with respect of mutations which would not change it.
    static Int64
    getUpdatedDataVersion(const DataPartPtr & part, std::unique_lock<std::mutex> & /* currently_processing_in_background_mutex_lock */);

    std::vector<PartVersionWithName> getSortedPartVersionsWithNames(std::unique_lock<std::mutex> & /* currently_processing_in_background_mutex_lock */) const;

    // Partition helpers
    void dropPartNoWaitNoThrow(const String & part_name) override;
    void dropPart(const String & part_name, bool detach, ContextPtr context) override;
    void dropPartition(const ASTPtr & partition, bool detach, ContextPtr context) override;
    void dropPartsImpl(DataPartsVector && parts_to_remove, bool detach);
    PartitionCommandsResultInfo attachPartition(const ASTPtr & partition, const StorageMetadataPtr & metadata_snapshot, bool part, ContextPtr context) override;

    void replacePartitionFrom(const StoragePtr & source_table, const ASTPtr & partition, bool replace, ContextPtr context) override;
    void movePartitionToTable(const StoragePtr & dest_table, const ASTPtr & partition, ContextPtr context) override;
    bool partIsAssignedToBackgroundOperation(const DataPartPtr & part) const override;

    void startBackgroundMovesIfNeeded() override;

    /// Attaches restored parts to the storage.
    void attachRestoredParts(MutableDataPartsVector && parts) override;

    std::unique_ptr<MergeTreeSettings> getDefaultSettings() const override;

    friend class UniqueMergeTreeSink;
    friend class MergeTreeData;
    friend class MergePlainUniqueMergeTreeTask;

protected:

    /** Attach the table with the appropriate name, along the appropriate path (with / at the end),
      *  (correctness of names and paths are not checked)
      *  consisting of the specified columns.
      *
      * See MergeTreeData constructor for comments on parameters.
      */
    StorageUniqueMergeTree(
        const StorageID & table_id_,
        const String & relative_data_path_,
        const StorageInMemoryMetadata & metadata,
        bool attach,
        ContextMutablePtr context_,
        const String & date_column_name,
        const MergingParams & merging_params_,
        std::unique_ptr<MergeTreeSettings> settings_,
        bool has_force_restore_data_flag);

    MutationCommands getFirstAlterMutationCommandsForPart(const DataPartPtr &) const override { return {}; }
};

}

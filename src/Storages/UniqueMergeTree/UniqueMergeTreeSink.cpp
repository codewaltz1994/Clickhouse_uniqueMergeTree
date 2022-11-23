#include <Interpreters/PartLog.h>
#include <Storages/MergeTree/MergeTreeDataPartInMemory.h>
#include <Storages/StorageUniqueMergeTree.h>
#include <Storages/UniqueMergeTree/UniqueMergeTreeSink.h>
#include <Storages/UniqueMergeTree/UniqueMergeTreeWriteState.h>


namespace DB
{

UniqueMergeTreeSink::~UniqueMergeTreeSink() = default;

/// TODO leefeng with __delete_op, the port header may do not match
UniqueMergeTreeSink::UniqueMergeTreeSink(
    StorageUniqueMergeTree & storage_, StorageMetadataPtr metadata_snapshot_, size_t max_parts_per_block_, ContextPtr context_)
    : SinkToStorage(storage_.getSampleBlockWithDeleteOp())
    , storage(storage_)
    , metadata_snapshot(metadata_snapshot_)
    , max_parts_per_block(max_parts_per_block_)
    , context(context_)
{
}

void UniqueMergeTreeSink::onStart()
{
    /// Only check "too many parts" before write,
    /// because interrupting long-running INSERT query in the middle is not convenient for users.
    storage.delayInsertOrThrowIfNeeded(nullptr, context);
}

void UniqueMergeTreeSink::onFinish()
{
    finishDelayedChunk();
}

struct UniqueMergeTreeSink::DelayedChunk
{
    struct Partition
    {
        MergeTreeDataWriter::TemporaryPart temp_part;
        UniqueMergeTreeWriteState write_state;
        UInt64 elapsed_ns;
        String block_dedup_token;
    };

    std::vector<Partition> partitions;
};


void UniqueMergeTreeSink::consume(Chunk chunk)
{
    auto block = getHeader().cloneWithColumns(chunk.detachColumns());
    auto storage_snapshot = storage.getStorageSnapshot(metadata_snapshot, context);

    auto part_blocks = storage.writer.splitBlockIntoParts(block, max_parts_per_block, metadata_snapshot, context);

    using DelayedPartitions = std::vector<UniqueMergeTreeSink::DelayedChunk::Partition>;
    DelayedPartitions partitions;

    const Settings & settings = context->getSettingsRef();
    size_t streams = 0;
    bool support_parallel_write = false;

    for (auto & current_block : part_blocks)
    {
        Stopwatch watch;
        String block_dedup_token;

        UniqueMergeTreeWriteState write_state;
        auto temp_part = storage.writer.writeTempPart(current_block, metadata_snapshot, context, write_state);

        /// This Context is used in select of PrimaryIndex update, such that to avoid dead lock betwwen insert
        /// and truncate/drop
        write_state.context = context;

        UInt64 elapsed_ns = watch.elapsed();

        /// If optimize_on_insert setting is true, current_block could become empty after merge
        /// and we didn't create part.
        /// May have delete
        if (!temp_part.part && !write_state.delete_key_column)
            continue;

        if (!support_parallel_write && temp_part.part->data_part_storage->supportParallelWrite())
            support_parallel_write = true;

        if (storage.getDeduplicationLog())
        {
            const String & dedup_token = settings.insert_deduplication_token;
            if (!dedup_token.empty())
            {
                /// multiple blocks can be inserted within the same insert query
                /// an ordinal number is added to dedup token to generate a distinctive block id for each block
                block_dedup_token = fmt::format("{}_{}", dedup_token, chunk_dedup_seqnum);
                ++chunk_dedup_seqnum;
            }
        }

        size_t max_insert_delayed_streams_for_parallel_write = DEFAULT_DELAYED_STREAMS_FOR_PARALLEL_WRITE;
        if (!support_parallel_write || settings.max_insert_delayed_streams_for_parallel_write.changed)
            max_insert_delayed_streams_for_parallel_write = settings.max_insert_delayed_streams_for_parallel_write;

        /// In case of too much columns/parts in block, flush explicitly.
        streams += temp_part.streams.size();
        if (streams > max_insert_delayed_streams_for_parallel_write)
        {
            finishDelayedChunk();
            delayed_chunk = std::make_unique<UniqueMergeTreeSink::DelayedChunk>();
            delayed_chunk->partitions = std::move(partitions);
            finishDelayedChunk();

            streams = 0;
            support_parallel_write = false;
            partitions = DelayedPartitions{};
        }

        partitions.emplace_back(UniqueMergeTreeSink::DelayedChunk::Partition{
            .temp_part = std::move(temp_part), .elapsed_ns = elapsed_ns, .block_dedup_token = std::move(block_dedup_token)});
    }

    finishDelayedChunk();
    delayed_chunk = std::make_unique<UniqueMergeTreeSink::DelayedChunk>();
    delayed_chunk->partitions = std::move(partitions);
}

void UniqueMergeTreeSink::finishDelayedChunk()
{
    if (!delayed_chunk)
        return;

    for (auto & partition : delayed_chunk->partitions)
    {
        /// No new part, just have delete
        if (!partition.temp_part.part && partition.write_state.delete_key_column)
        {
            if (partition.write_state.delete_key_column
                && storage.updatePrimaryIndexAndDeletes(
                    partition.write_state.partition,
                    partition.write_state.delete_key_column,
                    partition.write_state.min_key_values,
                    partition.write_state.max_key_values,
                    partition.write_state.context))
                storage.background_operations_assignee.trigger();
        }
        /// Have new part, may also have delete
        else
        {
            partition.temp_part.finalize();

            auto & part = partition.temp_part.part;

            bool added = false;

            /// It's important to create it outside of lock scope because
            /// otherwise it can lock parts in destructor and deadlock is possible.
            MergeTreeData::Transaction transaction(storage, context->getCurrentTransaction().get());
            {
                /// TODO leefeng, update primary index and delete bitmap should in here
                auto lock = storage.lockParts();
                storage.fillNewPartName(part, lock);

                auto * deduplication_log = storage.getDeduplicationLog();
                if (deduplication_log)
                {
                    const String block_id = part->getZeroLevelPartBlockID(partition.block_dedup_token);
                    auto res = deduplication_log->addPart(block_id, part->info);
                    if (!res.second)
                    {
                        ProfileEvents::increment(ProfileEvents::DuplicatedInsertedBlocks);
                        LOG_INFO(storage.log, "Block with ID {} already exists as part {}; ignoring it", block_id, res.first.getPartName());
                        continue;
                    }
                }

                added = storage.renameTempPartAndAdd(part, transaction, partition.temp_part.builder, lock);
                transaction.commit(&lock);
            }

            /// Part can be deduplicated, so increment counters and add to part log only if it's really added
            if (added)
            {
                PartLog::addNewPart(storage.getContext(), part, partition.elapsed_ns);
                storage.incrementInsertedPartsProfileEvent(part->getType());

                /// Initiate async merge - it will be done if it's good time for merge and if there are space in 'background_pool'.
                storage.background_operations_assignee.trigger();
            }
        }
    }

    delayed_chunk.reset();
}

}

#include <Storages/UniqueMergeTree/MergePlainUniqueMergeTreeTask.h>

#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/MergeTreeDataMergerMutator.h>
#include <Storages/StorageUniqueMergeTree.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int MERGE_ABORT;
}


StorageID MergePlainUniqueMergeTreeTask::getStorageID()
{
    return storage.getStorageID();
}

void MergePlainUniqueMergeTreeTask::onCompleted()
{
    bool delay = state == State::SUCCESS;
    task_result_callback(delay);
}


bool MergePlainUniqueMergeTreeTask::executeStep()
{
    /// Make out memory tracker a parent of current thread memory tracker
    MemoryTrackerThreadSwitcherPtr switcher;
    if (merge_list_entry)
        switcher = std::make_unique<MemoryTrackerThreadSwitcher>(*merge_list_entry);

    switch (state)
    {
        case State::NEED_PREPARE :
        {
            prepare();
            state = State::NEED_EXECUTE;
            return true;
        }
        case State::NEED_EXECUTE :
        {
            try
            {
                if (merge_task->execute())
                    return true;

                state = State::NEED_FINISH;
                return true;
            }
            catch (...)
            {
                write_part_log(ExecutionStatus::fromCurrentException());
                throw;
            }
        }
        case State::NEED_FINISH :
        {
            finish();

            state = State::SUCCESS;
            return false;
        }
        case State::SUCCESS:
        {
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Task with state SUCCESS mustn't be executed again");
        }
    }
    return false;
}


void MergePlainUniqueMergeTreeTask::prepare()
{
    future_part = merge_mutate_entry->future_part;
    stopwatch_ptr = std::make_unique<Stopwatch>();

    const Settings & settings = storage.getContext()->getSettingsRef();
    merge_list_entry = storage.getContext()->getMergeList().insert(
        storage.getStorageID(),
        future_part,
        settings);

    write_part_log = [this] (const ExecutionStatus & execution_status)
    {
        merge_task.reset();
        storage.writePartLog(
            PartLogElement::MERGE_PARTS,
            execution_status,
            stopwatch_ptr->elapsed(),
            future_part->name,
            new_part,
            future_part->parts,
            merge_list_entry.get());
    };

    merge_task = storage.merger_mutator.mergePartsToTemporaryPart(
        future_part,
        metadata_snapshot,
        merge_list_entry.get(),
        {} /* projection_merge_list_element */,
        table_lock_holder,
        time(nullptr),
        storage.getContext(),
        merge_mutate_entry->tagger->reserved_space,
        deduplicate,
        deduplicate_by_columns,
        storage.merging_params,
        nullptr,
        "",
        &storage);
}


void MergePlainUniqueMergeTreeTask::finish()
{
    new_part = merge_task->getFuture().get();

    auto & write_state = merge_task->getWriteState();

    storage.merger_mutator.renameMergedTemporaryPart(new_part, future_part->parts, storage, write_state, nullptr);
    storage.currently_processing_in_background_condition.notify_all();
    write_part_log({});
}

}

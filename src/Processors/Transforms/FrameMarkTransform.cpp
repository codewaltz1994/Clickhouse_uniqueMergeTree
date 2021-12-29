#include "FrameMarkTransform.h"
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnsNumber.h>
#include <Storages/WindowView/StorageWindowView.h>


namespace DB
{

FrameMarkTransform::FrameMarkTransform(const Block & header_, StorageWindowView & storage_, const String & window_column_name_)
    : ISimpleTransform(header_, header_, false), block_header(header_), storage(storage_), window_column_name(window_column_name_)
{
}

FrameMarkTransform::~FrameMarkTransform()
{
    if (max_watermark)
        storage.updateMaxFrame(max_watermark);
}

void FrameMarkTransform::transform(Chunk & chunk)
{
    auto num_rows = chunk.getNumRows();
    auto columns = chunk.detachColumns();

    auto column_window_idx = block_header.getPositionByName(window_column_name);
    const auto & window_column = columns[column_window_idx];
    const ColumnUInt32::Container & window_end_data = static_cast<const ColumnUInt32 &>(*window_column).getData();
    for (const auto & ts : window_end_data)
    {
        if (ts > max_watermark)
            max_watermark = ts;
    }

    chunk.setColumns(std::move(columns), num_rows);
}

}

#pragma once

#include <Processors/ISimpleTransform.h>

namespace DB
{

class StorageWindowView;

class FrameMarkTransform : public ISimpleTransform
{
public:
    FrameMarkTransform(
        const Block & header_, StorageWindowView & storage_, const String & window_column_name_);

    String getName() const override { return "FrameMarkTransform"; }

    ~FrameMarkTransform() override;

protected:
    void transform(Chunk & chunk) override;

    Block block_header;

    StorageWindowView & storage;
    String window_column_name;

    UInt32 max_watermark = 0;
};

}

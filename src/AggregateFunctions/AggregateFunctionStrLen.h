#pragma once

#include <type_traits>

#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>

#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>

#include <AggregateFunctions/IAggregateFunction.h>


namespace DB
{
struct AggregateFunctionStrLenData
{
    UInt64 sum{};

    void ALWAYS_INLINE add(UInt64 value) { sum += value; }

    void merge(const AggregateFunctionStrLenData & rhs) { sum += rhs.sum; }

    void write(WriteBuffer & buf) const
    {
        writeBinary(sum, buf);
    }

    void read(ReadBuffer & buf)
    {
        readBinary(sum, buf);
    }

    UInt64 get() const { return sum; }
};

template <typename Data>
class AggregateFunctionStrLen final : public IAggregateFunctionDataHelper<Data, AggregateFunctionStrLen<Data>>
{
public:
    String getName() const override { return "aggStrLen"; }

    AggregateFunctionStrLen(const DataTypes & argument_types_)
        : IAggregateFunctionDataHelper<Data, AggregateFunctionStrLen<Data>>(argument_types_, {})
    {}

    DataTypePtr getReturnType() const override { return std::make_shared<DataTypeUInt64>(); }

    void add(AggregateDataPtr place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        const auto & column = static_cast<const ColumnString &>(*columns[0]);
        const auto & offsets = column.getOffsets();
        this->data(place).add(offsets[row_num] - offsets[row_num - 1] - 1);
    }

    void merge(AggregateDataPtr place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr place, WriteBuffer & buf) const override
    {
        this->data(place).write(buf);
    }

    void deserialize(AggregateDataPtr place, ReadBuffer & buf, Arena *) const override
    {
        this->data(place).read(buf);
    }

    void insertResultInto(AggregateDataPtr place, IColumn & to, Arena *) const override
    {
        auto & column = static_cast<ColumnVector<UInt64> &>(to);
        column.getData().push_back(this->data(place).get());
    }
};

}

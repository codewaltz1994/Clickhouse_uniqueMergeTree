#pragma once

#include <AggregateFunctions/AggregateFunctionNull.h>

#include <Columns/ColumnArray.h>
#include <Columns/ColumnsNumber.h>

#include <Common/ArenaAllocator.h>
#include <Common/assert_cast.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypesNumber.h>

#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>

#include <unordered_set>

#include <rpc/client.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int TOO_LARGE_ARRAY_SIZE;
}

struct AggregateFunctionAccData
{
    constexpr static size_t MAX_ARRAY_SIZE = 0xFFFFFF;
    using Array = PODArray<UInt64, 64>;

    Array array;

    void add(UInt64 data) { array.emplace_back(data); }

    void merge(const AggregateFunctionAccData & other)
    {
        if (other.array.empty())
            return;

        array.insert(std::begin(other.array), std::end(other.array));
    }

    void serialize(WriteBuffer & buf) const
    {
        writeBinary(array.size(), buf);

        for (auto data : array)
        {
            writeBinary(data, buf);
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        size_t size;
        readBinary(size, buf);

        if (unlikely(size > MAX_ARRAY_SIZE))
            throw Exception("Too large array size", ErrorCodes::TOO_LARGE_ARRAY_SIZE);

        array.reserve(size);

        UInt64 data;
        for (size_t i = 0; i < size; ++i)
        {
            readBinary(data, buf);
            array.emplace_back(data);
        }
    }
};

template <typename Data>
class AggregateFunctionAcc final : public IAggregateFunctionDataHelper<Data, AggregateFunctionAcc<Data>>
{
private:

public:
    String getName() const override { return "syncFinal"; }

    explicit AggregateFunctionAcc(const DataTypes & arguments)
        : IAggregateFunctionDataHelper<Data, AggregateFunctionAcc<Data>>(arguments, {})
    {
    }

    DataTypePtr getReturnType() const override { return std::make_shared<DataTypeArray>(std::make_shared<DataTypeFloat64>()); }

    bool allocatesMemoryInArena() const override { return false; }

    AggregateFunctionPtr getOwnNullAdapter(
        const AggregateFunctionPtr & nested_function,
        const DataTypes & arguments,
        const Array & params,
        const AggregateFunctionProperties & /*properties*/) const override
    {
        return std::make_shared<AggregateFunctionNullVariadic<false, false, false>>(nested_function, arguments, params);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, const size_t row_num, Arena *) const override
    {
        auto data = assert_cast<const ColumnVector<UInt64> *>(columns[0])->getData()[row_num];
        this->data(place).add(data);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t>) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t>, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        rpc::client client("10.60.1.80", 8090);
        auto res = client.call("syncFinal", this->data(place).array[0], this->data(place).array[1]).template as<std::vector<float>>();
        Array array(res.begin(), res.end());
        assert_cast<ColumnArray &>(to).insert(array);
    }
};

}

#pragma once

#include <AggregateFunctions/AggregateFunctionNull.h>

#include <Columns/ColumnArray.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>

#include <Common/ArenaAllocator.h>
#include <Common/assert_cast.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypesNumber.h>

#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>

#include <rpc/client.h>

#include <unordered_set>
#include <sys/ipc.h>
#include <sys/shm.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int TOO_LARGE_ARRAY_SIZE;
    extern const int CANNOT_ALLOCATE_MEMORY;
}

struct AggregateFunctionWLData
{
    constexpr static size_t MAX_ARRAY_SIZE = 0xFFFFFF;
    using Array = PODArray<String, 64>;

    Array array;

    void add(String data) { array.emplace_back(data); }

    void merge(const AggregateFunctionWLData & other)
    {
        if (other.array.empty())
            return;

        array.insert(std::begin(other.array), std::end(other.array));
    }

    void serialize(WriteBuffer & buf) const
    {
        writeBinary(array.size(), buf);

        for (const auto & data : array)
        {
            writeStringBinary(data, buf);
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        size_t size;
        readBinary(size, buf);

        if (unlikely(size > MAX_ARRAY_SIZE))
            throw Exception("Too large array size", ErrorCodes::TOO_LARGE_ARRAY_SIZE);

        array.reserve(size);

        String data;
        for (size_t i = 0; i < size; ++i)
        {
            readStringBinary(data, buf);
            array.emplace_back(data);
        }
    }
};

template <typename T, typename Data>
class AggregateFunctionWL final : public IAggregateFunctionDataHelper<Data, AggregateFunctionWL<T, Data>>
{
private:

public:
    String getName() const override { return "WL"; }

    explicit AggregateFunctionWL(const DataTypes & arguments)
        : IAggregateFunctionDataHelper<Data, AggregateFunctionWL<T, Data>>(arguments, {})
    {
    }

    DataTypePtr getReturnType() const override { return std::make_shared<DataTypeArray>(std::make_shared<DataTypeFloat32>()); }

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
        Field f;
        assert_cast<const ColumnString *>(columns[0])->get(row_num, f);
        this->data(place).add(f.get<String>());
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t>) const override
    {
        this->data(place).serialize(buf);
    }

    void
    deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t>, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        rpc::client client("127.0.0.1", 8080);

        int magic = 7;
        std::string file_name = "wl_data";

        key_t key = ftok(file_name.c_str(), magic);
        auto data_size = this->data(place).array[0].size();

        int shmid = shmget(key, data_size, 0666 | IPC_CREAT);
        if (shmid == -1)
        {
            throw Exception("Create shared memory faild", ErrorCodes::CANNOT_ALLOCATE_MEMORY);
        }
        /// attach to shared memory
        char * str = reinterpret_cast<char *>(shmat(shmid, reinterpret_cast<void *>(0), 0));
        memcpy(str, this->data(place).array[0].data(), data_size);
        /// detach from shared memory
        shmdt(str);

        auto vec = client.call("wl", file_name, magic, data_size).template as<std::vector<float>>();
        Array array(vec.begin(), vec.end());
        assert_cast<ColumnArray &>(to).insert(array);
        shmctl(shmid, IPC_RMID, nullptr);
    }
};

}

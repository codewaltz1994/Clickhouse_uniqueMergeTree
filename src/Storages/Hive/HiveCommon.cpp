#include <Storages/Hive/HiveCommon.h>

#if USE_HIVE

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TSocket.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int NO_HIVEMETASTORE;
    extern const int BAD_ARGUMENTS;
}

bool HiveMetastoreClient::shouldUpdateTableMetadata(
    const String & db_name, const String & table_name, const std::vector<Apache::Hadoop::Hive::Partition> & partitions)
{
    String cache_key = getCacheKey(db_name, table_name);
    HiveTableMetadataPtr metadata = table_metadata_cache.get(cache_key);
    if (!metadata)
        return true;

    auto old_partiton_infos = metadata->getPartitionInfos();
    if (old_partiton_infos.size() != partitions.size())
        return true;

    for (const auto & partition : partitions)
    {
        auto it = old_partiton_infos.find(partition.sd.location);
        if (it == old_partiton_infos.end())
            return true;

        const auto & old_partition_info = it->second;
        if (!old_partition_info.haveSameParameters(partition))
            return true;
    }
    return false;
}

HiveMetastoreClient::HiveTableMetadataPtr HiveMetastoreClient::getTableMetadata(const String & db_name, const String & table_name)
{
    LOG_TRACE(log, "Get table metadata for {}.{}", db_name, table_name);
    std::lock_guard lock{mutex};

    auto table = std::make_shared<Apache::Hadoop::Hive::Table>();
    std::vector<Apache::Hadoop::Hive::Partition> partitions;
    try
    {
        client->get_table(*table, db_name, table_name);

        /// Query the latest partition info to check new change.
        client->get_partitions(partitions, db_name, table_name, -1);
    }
    catch (apache::thrift::transport::TTransportException & e)
    {
        setExpired();
        throw Exception(ErrorCodes::NO_HIVEMETASTORE, "Hive Metastore expired because {}", String(e.what()));
    }

    bool update_cache = shouldUpdateTableMetadata(db_name, table_name, partitions);
    String cache_key = getCacheKey(db_name, table_name);

    HiveTableMetadataPtr metadata = table_metadata_cache.get(cache_key);

    if (update_cache)
    {
        LOG_INFO(log, "Reload hive partition metadata info for {}.{}", db_name, table_name);

        /// Generate partition infos from partitions and old partition infos(if exists).
        std::map<String, PartitionInfo> new_partition_infos;
        if (metadata)
        {
            auto & old_partiton_infos = metadata->getPartitionInfos();
            for (const auto & partition : partitions)
            {
                auto it = old_partiton_infos.find(partition.sd.location);
                if (it == old_partiton_infos.end() || !it->second.haveSameParameters(partition) || !it->second.initialized)
                {
                    new_partition_infos.emplace(partition.sd.location, PartitionInfo(partition));
                    continue;
                }
                else
                {
                    PartitionInfo new_partition_info(partition);
                    new_partition_info.files = std::move(it->second.files);
                    new_partition_info.initialized = true;
                }
            }
        }
        else
        {
            for (const auto & partition : partitions)
                new_partition_infos.emplace(partition.sd.location, PartitionInfo(partition));
        }

        metadata = std::make_shared<HiveMetastoreClient::HiveTableMetadata>(
            db_name, table_name, table, std::move(new_partition_infos), getContext());
        table_metadata_cache.set(cache_key, metadata);
    }
    return metadata;
}

void HiveMetastoreClient::clearTableMetadata(const String & db_name, const String & table_name)
{
    String cache_key = getCacheKey(db_name, table_name);

    std::lock_guard lock{mutex};
    HiveTableMetadataPtr metadata = table_metadata_cache.get(cache_key);
    if (metadata)
        table_metadata_cache.remove(cache_key);
}

void HiveMetastoreClient::setClient(std::shared_ptr<Apache::Hadoop::Hive::ThriftHiveMetastoreClient> client_)
{
    std::lock_guard lock{mutex};
    client = client_;
    clearExpired();
}

bool HiveMetastoreClient::PartitionInfo::haveSameParameters(const Apache::Hadoop::Hive::Partition & other) const
{
    /// Parameters include keys:numRows,numFiles,rawDataSize,totalSize,transient_lastDdlTime
    auto it1 = partition.parameters.cbegin();
    auto it2 = other.parameters.cbegin();
    for (; it1 != partition.parameters.cend() && it2 != other.parameters.cend(); ++it1, ++it2)
    {
        if (it1->first != it2->first || it1->second != it2->second)
            return false;
    }
    return (it1 == partition.parameters.cend() && it2 == other.parameters.cend());
}

std::vector<Apache::Hadoop::Hive::Partition> HiveMetastoreClient::HiveTableMetadata::getPartitions() const
{
    std::vector<Apache::Hadoop::Hive::Partition> result;

    std::lock_guard lock{mutex};
    for (const auto & partition_info : partition_infos)
        result.emplace_back(partition_info.second.partition);
    return result;
}

std::vector<HiveMetastoreClient::FileInfo> HiveMetastoreClient::HiveTableMetadata::getFilesByLocation(const HDFSFSPtr & fs, const String & location)
{
    LOG_TRACE(log, "List directory {}", location);
    std::map<String, PartitionInfo>::iterator it;
    if (!empty_partition_keys)
    {
        std::lock_guard lock{mutex};
        it = partition_infos.find(location);
        if (it == partition_infos.end())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Invalid location {}", location);

        if (it->second.initialized)
        {
            LOG_TRACE(log, "Get {} files under directory {}", it->second.files.size(), location);
            return it->second.files;
        }
    }

    Poco::URI location_uri(location);
    HDFSFileInfo ls;
    ls.file_info = hdfsListDirectory(fs.get(), location_uri.getPath().c_str(), &ls.length);
    std::vector<FileInfo> result;
    for (int i = 0; i < ls.length; ++i)
    {
        auto & file_info = ls.file_info[i];
        if (file_info.mKind != 'D' && file_info.mSize > 0)
            result.emplace_back(String(file_info.mName), file_info.mLastMod, file_info.mSize);
    }

    if (!empty_partition_keys)
    {
        std::lock_guard lock{mutex};
        it = partition_infos.find(location);
        if (it == partition_infos.end())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Invalid location {}", location);

        it->second.files = result;
        it->second.initialized = true;
    }
    LOG_TRACE(log, "Get {} files under directory {}", result.size(), location);
    return result;
}

HiveMetastoreClientFactory & HiveMetastoreClientFactory::instance()
{
    static HiveMetastoreClientFactory factory;
    return factory;
}

HiveMetastoreClientPtr HiveMetastoreClientFactory::getOrCreate(const String & name, ContextPtr context)
{
    using namespace apache::thrift;
    using namespace apache::thrift::protocol;
    using namespace apache::thrift::transport;
    using namespace Apache::Hadoop::Hive;

    std::lock_guard lock(mutex);
    auto it = clients.find(name);
    if (it == clients.end() || it->second->isExpired())
    {
        /// Connect to hive metastore
        Poco::URI hive_metastore_url(name);
        const auto & host = hive_metastore_url.getHost();
        auto port = hive_metastore_url.getPort();

        std::shared_ptr<TSocket> socket = std::make_shared<TSocket>(host, port);
        socket->setKeepAlive(true);
        socket->setConnTimeout(conn_timeout_ms);
        socket->setRecvTimeout(recv_timeout_ms);
        socket->setSendTimeout(send_timeout_ms);
        std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
        std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
        std::shared_ptr<ThriftHiveMetastoreClient> thrift_client = std::make_shared<ThriftHiveMetastoreClient>(protocol);
        try
        {
            transport->open();
        }
        catch (TException & tx)
        {
            throw Exception("connect to hive metastore:" + name + " failed." + tx.what(), ErrorCodes::BAD_ARGUMENTS);
        }

        if (it == clients.end())
        {
            HiveMetastoreClientPtr client = std::make_shared<HiveMetastoreClient>(std::move(thrift_client), context);
            clients[name] = client;
            return client;
        }
        else
        {
            it->second->setClient(std::move(thrift_client));
            return it->second;
        }
    }
    return it->second;
}

}
#endif

#pragma once

#include <Common/config.h>

#if USE_AWS_S3

#    include <atomic> // for std::atomic<size_t>
#    include <mutex>

#    include "ProxyConfiguration.h"

namespace DB::S3
{

class ProxyPalorisConfiguration : public ProxyConfiguration
{
public:
    enum InstanceCode
    {
        NO_CHANGE = 200001,
        HAS_CHAGE = 200000,
    };

    struct Proxy
    {
        Poco::URI host;
        bool valid = true;
        explicit Proxy(const std::string & host_) : host(host_) { }
    };

    explicit ProxyPalorisConfiguration(
        const std::string & paloris_host_, const std::string & service_name_, const std::string & service_namespace_, unsigned cache_ttl_);

    ClientConfigurationPerRequest getConfiguration(const Aws::Http::HttpRequest & request) override;
    void errorReport(const ClientConfigurationPerRequest &) override;


private:
    void initialize();

    /// create JSON string for POST
    std::string createContentJsonString() const;

    void updateAvaliableHosts();

    /// Arguments for http request to Paloris
    std::string paloris_host; //"http://polaris-discover.oa.com:8080/v1/Discover";

    std::string service_name;
    std::string service_namespace;
    /// The revision of the last request to Paloris, so it is empty as the first
    std::string revision{};

    std::mutex cache_mutex;

    int valid_host_cnt = 0;

    /// List of cached hosts.
    std::vector<Proxy> cached_hosts;

    const std::chrono::seconds cache_ttl{0};
    std::chrono::time_point<std::chrono::system_clock> cache_timestamp;
    std::atomic<size_t> access_counter{0};
};

}

#endif

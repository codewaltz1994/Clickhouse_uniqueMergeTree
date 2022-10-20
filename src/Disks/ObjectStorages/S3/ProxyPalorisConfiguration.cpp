#include "ProxyPalorisConfiguration.h"

#if USE_AWS_S3

#    include <utility>
#    include <IO/HTTPCommon.h>
#    include "Poco/StreamCopier.h"
#    include <Poco/JSON/Array.h>
#    include <Poco/JSON/Object.h>
#    include <Poco/JSON/Parser.h>
#    include <Poco/Net/HTTPRequest.h>
#    include <Poco/Net/HTTPResponse.h>
#    include <Common/logger_useful.h>

namespace DB::S3
{

ProxyPalorisConfiguration::ProxyPalorisConfiguration(
    const std::string & paloris_host_, const std::string & service_name_, const std::string & service_namespace_, unsigned cache_ttl_)
    : paloris_host(paloris_host_), service_name(service_name_), service_namespace(service_namespace_), cache_ttl(cache_ttl_)
{
    initialize();
}

void ProxyPalorisConfiguration::initialize()
{
    updateAvaliableHosts();
    cache_timestamp = std::chrono::system_clock::now();
    valid_host_cnt = cached_hosts.size();
}

ClientConfigurationPerRequest ProxyPalorisConfiguration::getConfiguration(const Aws::Http::HttpRequest &)
{
    LOG_DEBUG(&Poco::Logger::get("AWSClient"), "Obtain proxy using paloris: {}", paloris_host);

    std::unique_lock lock(cache_mutex);

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();

    /// If there is no valid hosts or cache ttl exceed, update avaliable hosts
    if (!valid_host_cnt || (cache_ttl.count() && now > cache_timestamp + cache_ttl))
    {
        updateAvaliableHosts();
        cache_timestamp = std::chrono::system_clock::now();
        valid_host_cnt = cached_hosts.size();
    }

    ClientConfigurationPerRequest cfg;

    if (!valid_host_cnt)
    {
        throw Exception("There is no valid host in paloris proxy: " + paloris_host, ErrorCodes::BAD_ARGUMENTS);
    }

    /// Avoid atomic increment if number of proxies is 1.
    size_t index = cached_hosts.size() > 1 ? (access_counter++) % cached_hosts.size() : 0;
    /// Find next valid host from last access location
    while (!cached_hosts[index].valid)
    {
        index = cached_hosts.size() > 1 ? (access_counter++) % cached_hosts.size() : 0;
    }

    cfg.proxy_scheme = Aws::Http::SchemeMapper::FromString(cached_hosts[index].host.getScheme().c_str());
    cfg.proxy_host = cached_hosts[index].host.getHost();
    cfg.proxy_port = cached_hosts[index].host.getPort();

    LOG_DEBUG(
        &Poco::Logger::get("AWSClient"),
        "Use cached paloris proxy: {}://{}:{}",
        Aws::Http::SchemeMapper::ToString(cfg.proxy_scheme),
        cfg.proxy_host,
        cfg.proxy_port);
    return cfg;
}

void ProxyPalorisConfiguration::updateAvaliableHosts()
{
    ConnectionTimeouts timeouts(
        Poco::Timespan(1000000), /// Connection timeout.
        Poco::Timespan(1000000), /// Send timeout.
        Poco::Timespan(1000000) /// Receive timeout.
    );
    try
    {
        /// Send http POST to paloris host to get all instances info
        Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, paloris_host, Poco::Net::HTTPRequest::HTTP_1_1);

        request.setContentType("application/json");
        auto body = createContentJsonString();
        request.setContentLength(body.size());

        auto http_session = makeHTTPSession(Poco::URI(paloris_host), timeouts, false);
        http_session->sendRequest(request) << body;

        Poco::Net::HTTPResponse response;
        auto & response_body_stream = http_session->receiveResponse(response);

        if (response.getStatus() != Poco::Net::HTTPResponse::HTTP_OK)
            throw Exception("Paloris http request returned not OK status: " + response.getReason(), ErrorCodes::BAD_ARGUMENTS);

        std::string instances_info;
        Poco::StreamCopier::copyToString(response_body_stream, instances_info);

        /// Responce from paloris is a JSON string, need to exact info from it.
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(instances_info);
        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        size_t code = object->getValue<size_t>("code");

        /// Instances has no change, does not need to update avaliable hosts
        if (code == InstanceCode::NO_CHANGE)
        {
            LOG_DEBUG(
                &Poco::Logger::get("AWSClient"),
                "Paloris return code {}, service instances has no change since last call: {}",
                InstanceCode::NO_CHANGE,
                paloris_host);
        }
        /// Instances has change, update cached hosts
        else if (code == InstanceCode::HAS_CHAGE)
        {
            Poco::Dynamic::Var service = object->get("service");
            Poco::JSON::Object::Ptr sub_object = service.extract<Poco::JSON::Object::Ptr>();

            /// Update revision, will be used in next paloris call
            revision = sub_object->getValue<String>("revision");

            Poco::Dynamic::Var instances = object->get("instances");
            Poco::JSON::Array::Ptr instances_array = instances.extract<Poco::JSON::Array::Ptr>();

            std::vector<Proxy> new_hosts;
            for (size_t i = 0; i < instances_array->size(); ++i)
            {
                Poco::JSON::Object::Ptr instance = instances_array->getObject(i);
                auto host = instance->getValue<String>("host");
                /// Port does not need, Poco::URI can infer by schema, 80 for http
                // auto port = instance->getValue<int>("port");
                auto healthy = instance->getValue<String>("healthy");
                auto weight = instance->getValue<UInt32>("weight");

                if (healthy == "true" && weight)
                {
                    new_hosts.emplace_back("http://" + host);
                }
            }

            if (new_hosts.empty())
                throw Exception(
                    "Paloris http request get empty healthy hosts: " + paloris_host + ", service_name: " + service_name
                        + ", service_namespace: " + service_namespace,
                    ErrorCodes::BAD_ARGUMENTS);

            cached_hosts.swap(new_hosts);
        }
        else
        {
            throw Exception("Unknown code return from Paloris: " + paloris_host, ErrorCodes::BAD_ARGUMENTS);
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

std::string ProxyPalorisConfiguration::createContentJsonString() const
{
    return R"({"type": 1, "service": {"name": ")" + service_name + R"(", "namespace": ")" + service_namespace + R"(", "revision" : ")"
        + revision + R"("}})";
}

void ProxyPalorisConfiguration::errorReport(const ClientConfigurationPerRequest & config)
{
    if (config.proxy_host.empty())
        return;

    std::unique_lock lock(cache_mutex);

    if (!cache_ttl.count() || !valid_host_cnt)
        return;

    /// Invalidate cached proxy when got error with this proxy
    for (auto & proxy : cached_hosts)
    {
        if (proxy.host.getScheme() == Aws::Http::SchemeMapper::ToString(config.proxy_scheme) && proxy.host.getHost() == config.proxy_host
            && proxy.host.getPort() == config.proxy_port)
        {
            if (proxy.valid)
            {
                proxy.valid = false;
                --valid_host_cnt;
            }
        }
    }
}

}

#endif

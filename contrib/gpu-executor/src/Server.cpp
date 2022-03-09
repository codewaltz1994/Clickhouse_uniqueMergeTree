#include "Server.h"

#include <iostream>

Server::Server()
{
    func = std::make_unique<Functions>();
    rpc_server = std::make_unique<rpc::server>(8090);
}

void Server::init()
{
    /// bind function to rpc server
#if 0
    rpc_server->bind("init", [&]() { algo->init(); });
    rpc_server->bind("get_device_num", [&]() -> int32_t { return algo->getDeviceNum(); });
    rpc_server->bind("echo_data_1", [&](const std::string file_name, int magic, size_t data_size) -> uint64_t {
        return algo->echo_data_1(file_name, magic, data_size);
    });

    rpc_server->bind("wl_1", [&](const std::string file_name, int magic, size_t data_size) -> std::vector<float> {
        return algo->WL_1(file_name, magic, data_size);
    });

    rpc_server->bind("wl_2", [&](const std::string file_name, int magic, size_t data_size) -> std::vector<float> {
        return algo->WL_2(file_name, magic, data_size);
    });

    rpc_server->bind("dbf_exec_1", [&](uint64_t d_echo_data_1) -> uint64_t { return algo->dbf_exec_1(d_echo_data_1); });

    rpc_server->bind("pc_exec_1", [&](uint64_t dbf_out_1) -> uint64_t { return algo->pc_exec_1(dbf_out_1); });

    rpc_server->bind("mtd_exec_1", [&](uint64_t pc_out_1) -> uint64_t { return algo->mtd_exec_1(pc_out_1); });

    rpc_server->bind("pc_later_1", [&](uint64_t d_echo_data_1) -> uint64_t { return algo->pc_later_1(d_echo_data_1); });

    rpc_server->bind("second_mtd_exec_1", [&](uint64_t pc_later_out_1) -> uint64_t { return algo->second_mtd_exec_1(pc_later_out_1); });

    rpc_server->bind("echo_data_2", [&](const std::string file_name, int magic, size_t data_size) -> uint64_t {
        return algo->echo_data_2(file_name, magic, data_size);
    });

    rpc_server->bind("dbf_exec_2", [&](uint64_t d_echo_data_2) -> uint64_t { return algo->dbf_exec_2(d_echo_data_2); });

    rpc_server->bind("pc_exec_2", [&](uint64_t dbf_out_2) -> uint64_t { return algo->pc_exec_2(dbf_out_2); });

    rpc_server->bind("mtd_exec_2", [&](uint64_t pc_out_2) -> uint64_t { return algo->mtd_exec_2(pc_out_2); });

    rpc_server->bind("first_cfar_exec_2", [&](uint64_t some_addr) -> uint64_t { return algo->first_cfar_exec_2(some_addr); });

    rpc_server->bind("pc_later_2", [&](uint64_t d_echo_data_2) -> uint64_t { return algo->pc_later_2(d_echo_data_2); });

    rpc_server->bind("second_mtd_exec_2", [&](uint64_t pc_later_out_2) -> uint64_t { return algo->second_mtd_exec_2(pc_later_out_2); });
    rpc_server->bind("second_dbf_exec_2", [&](uint64_t some_addr) -> std::vector<float_t> { return algo->second_dbf_exec_2(some_addr); });
    rpc_server->bind("acc", [&]() -> std::vector<unsigned long> { return algo->acc(); });
#endif
}

void Server::run()
{
    rpc_server->async_run(8);
    std::cin.ignore();
}

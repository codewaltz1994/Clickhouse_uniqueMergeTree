#pragma once

#include <memory>
#include <rpc/server.h>
#include "UserDefinedFunctions.h"

class Server
{
public:
    Server();
    Server(const Server &) = delete;
    Server & operator=(const Server &) = delete;
    void init();
    void run();

private:
    std::unique_ptr<Functions> func;
    std::unique_ptr<rpc::server> rpc_server;
};

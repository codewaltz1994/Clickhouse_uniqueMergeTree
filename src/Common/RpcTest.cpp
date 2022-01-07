#include <rpc/client.h>
#include <rpc/server.h>

namespace DB
{
void createClient()
{
    rpc::client client("127.0.0.1", 8080);
}
}

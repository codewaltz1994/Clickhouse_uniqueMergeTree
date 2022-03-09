#include <ctime>
#include <iostream>
#include <string>

#include <Stopwatch.h>
#include <CpuLZ4Compressor.h>
#include <zmq.hpp>

int main()
{
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REQ (request) socket and connect to interface
    zmq::socket_t socket{context, zmq::socket_type::req};
    socket.connect("tcp://10.208.61.9:5555");

    // set up some static data to send
    const std::string data{"request data"};

    for (auto request_num = 0; request_num < 10; ++request_num)
    {
        // send the request message
        std::cout << "Sending request data " << request_num << "..." << std::endl;
        socket.send(zmq::buffer(data), zmq::send_flags::none);

        // wait for reply from server
        zmq::message_t reply{};
        socket.recv(reply, zmq::recv_flags::none);

        std::string s;
        s.resize(1024 * 1024 * 1024);
        Stopwatch sw;
        auto return_uncompressed_size = CpuLZ4::decompress((char *)reply.data(), reply.size(), (char *)s.data(), 1024 * 1024 * 1024);
        auto seconds = sw.elapsedSeconds();
        std::cout << "time in decompress " << seconds << " seconds" << std::endl;

        auto recv_time = time(NULL);
        std::cout << "receive time: " << recv_time << std::endl;

        std::cout << "Received data " << request_num << std::endl;
        std::cout << std::endl;
    }

    return 0;
}


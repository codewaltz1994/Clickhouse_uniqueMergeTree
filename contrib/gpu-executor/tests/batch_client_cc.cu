#include <ctime>
#include <iostream>
#include <string>

#include <Stopwatch.h>
#include <CpuLZ4Compressor.h>
#include <zmq.hpp>

size_t wordCount(const std::string & s)
{
    size_t word_count = 0;
    for (auto c : s)
    {
        if (c == ' ' || c == '\t' || c == '\n')
            ++word_count;
    }
    return word_count;
}

int main()
{
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REQ (request) socket and connect to interface
    zmq::socket_t socket{context, zmq::socket_type::req};
    socket.connect("tcp://10.208.61.9:5555");

    // set up some static data to send
    const std::string data{"request data"};

    for (size_t batch_size = 1; batch_size <= 20000; batch_size <<= 1)
    {
        for (size_t request_num = 0; request_num < 10; ++request_num)
        {
            // send the request message
            std::cout << "Sending request data " << request_num << "..." << std::endl;
            socket.send(zmq::buffer(data), zmq::send_flags::none);

            // wait for reply from server
            zmq::message_t reply{}, uncompressed_size{};
            socket.recv(reply, zmq::recv_flags::none);
            socket.recv(uncompressed_size, zmq::recv_flags::none);

            std::string s;
            size_t uncompressed_data_size = std::stoi((char *)uncompressed_size.data());
            s.resize(uncompressed_data_size);

            Stopwatch sw;
            auto return_uncompressed_size
                = CpuLZ4::decompress((char *)reply.data(), reply.size(), (char *)s.data(), uncompressed_data_size);
            auto seconds = sw.elapsedMilliseconds();
            std::cout << "time in decompress " << seconds << " milli seconds" << std::endl;

            auto words = wordCount(s);

            std::cout << "Received data " << request_num << ", word nums = " << words << std::endl;
            std::cout << std::endl;
        }
    }


    return 0;
}


#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <Stopwatch.h>
#include <CpuLZ4Compressor.h>

#include <zmq.hpp>

// 1GB
constexpr size_t data_size = 1024 * 1024 * 1024;

std::string getData()
{
    std::ifstream is("/data/fenglv/final/gpu-executor/data/ol_cdump.json");
    std::string data;
    data.resize(data_size);
    is.read((char *)data.data(), data_size);
    return data;
}

int main()
{
    using namespace std::chrono_literals;

    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REP (reply) socket and bind to interface
    zmq::socket_t socket{context, zmq::socket_type::rep};
    socket.bind("tcp://*:5555");

    // prepare some static data for responses
    const std::string data{"World"};

    auto message_data = getData();

    for (;;)
    {
        zmq::message_t request;

        // receive a request from client
        socket.recv(request, zmq::recv_flags::none);

        auto send_time = time(NULL);
        std::cout << "sent time: " << send_time << std::endl;
        size_t max_comressed_size = CpuLZ4::getMaxCompressedDataSize(data_size);
        std::string compress_data;
        compress_data.resize(max_comressed_size);
        Stopwatch sw;
        size_t compressed_size = CpuLZ4::compress((char *)message_data.data(), data_size, (char *)compress_data.data());
        std::cout << "uncompressed_size = " << data_size / 1024.0 / 1024.0 << " MB, compressed_size = " << compressed_size / 1024.0 / 1024.0
                  << " MB" << std::endl;
        auto seconds = sw.elapsedSeconds();
        std::cout << "time in compress " << seconds << " seconds" << std::endl;
        std::cout << "Compress ratio = " << (double)data_size / compressed_size << std::endl;
        //std::string s = std::string("sendTime:") + std::to_string(send_time);
        //socket.set(zmq::sockopt::metadata, zmq::str_buffer(s);

        // simulate work
        // std::this_thread::sleep_for(1s);

        // send the reply to the client
        socket.send(zmq::buffer(compress_data.data(), compressed_size), zmq::send_flags::none);
    }

    return 0;
}


#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <Stopwatch.h>
#include <CpuLZ4Compressor.h>

#include <zmq.hpp>

// 1GB

std::string getData(size_t batch_size)
{
    std::ifstream is("/data/fenglv/final/gpu-executor/data/ol_cdump.json");
    std::string data;
    std::string s;
    for (size_t i = 0; i < batch_size; ++i)
    {
        std::getline(is, s);
        data += s;
    }
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

    size_t batch_size = 1;
    size_t new_batch_size = 1;
    auto message_data = getData(batch_size);
    size_t send_nums = 0;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::vector<double> times;
    for (;;)
    {
        zmq::message_t request;

        if (new_batch_size != batch_size)
        {
            std::cout << "Time for batch size " << batch_size << " = " << std::accumulate(times.begin(), times.end(), 0.0) / 10.0
                      << std::endl
                      << std::endl;
            times.clear();
            message_data = getData(new_batch_size);
            batch_size = new_batch_size;
        }
        // receive a request from client
        socket.recv(request, zmq::recv_flags::none);

        if (send_nums)
        {
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            size_t micro_seconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            times.emplace_back(double(micro_seconds) / 1000);
        }

        start = std::chrono::high_resolution_clock::now();

        // send the reply to the client
        socket.send(zmq::buffer(message_data), zmq::send_flags::sndmore);
        socket.send(zmq::buffer(std::to_string(message_data.size())), zmq::send_flags::none);
        ++send_nums;

        if (send_nums % 10 == 0)
            new_batch_size <<= 1;
    }

    return 0;
}


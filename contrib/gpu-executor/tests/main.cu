#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <string>
#include <stdio.h>
#include <nvcomp.h>
#include "Server.h"
#include "device_launch_parameters.h"

#if 0
int runFunc()
{
    int pointerNum = 10;
    int serverSize = 1;
    int JDSize = 2;
    int JDArray[] = {3,4};
    int mode = 1;
    int recv_mode = 1;
    int CPI_pipeline = 1;
    int sleepCount = 1;
    string dhsm_url = "10.61.2.88";
    AlgorithmClass hd(pointerNum,
            JDArray,
            serverSize,
            JDSize,
            mode,
            recv_mode,
            CPI_pipeline,
             sleepCount, dhsm_url);

    //进行服务器的参数初始化及节点帧时序同步至起始时序
    hd.Malloc_para();
    // this_thread::sleep_for(std::chrono::seconds(sleepCount));

    // int initflag = 0;
    hd.Init();
    // while(!initflag){
    // }

    //初始化完成后，启动雷达数据收数并存入内存块线程
    thread t1(&AlgorithmClass::CollectData_gpu,&hd, 1);
    t1.join();

    thread t2(&AlgorithmClass::CollectData_gpu,&hd, 2);
    t2.join();
    
    uint64_t start_ = timeSinceEpochMillisec();

    std::cout << "\033[31mAAAAAAAAAAAAAAAA start_: " << start_ << "\033[39m" << std::endl;

    //启动对收数后内存块中的雷达数据处理线程
    // thread t3(&AlgorithmClass::ProcessData_gpu1,&hd);
    // thread t4(&AlgorithmClass::ProcessData_gpu2,&hd);

    auto f3 = [&](){
        hd.ProcessData_gpu1();
    };
    auto f4 = [&](){
        hd.ProcessData_gpu2();
    };

    thread t3(f3);
    thread t4(f4);

    t3.join();
    t4.join();

    uint64_t end_ = timeSinceEpochMillisec();

    std::cout << "\033[31mAAAAAAAAAAAAAAAA end_: " << end_ << "\033[39m" << std::endl;

    std::cout << "\033[31mAAAAAAAAAAAAAAAA time: " << (end_ - start_) << "\033[39m" << std::endl;

    // hd.CollectData_gpu1();
    // hd.CollectData_gpu2();
// 
    // hd.ProcessData_gpu1();
    // hd.ProcessData_gpu2();

    // //当存在丢帧时重启
    // while(!hd.CollectErrorDone && initflag){
    // }
    // hd->~AlgorithmClass();
    return 1;
}
#endif

int main()
{
	Server server;
	std::cout << "Create server listenning to port 8090..."<<std::endl;
	server.init();
	std::cout << "Starting run server...";
	server.run();
}

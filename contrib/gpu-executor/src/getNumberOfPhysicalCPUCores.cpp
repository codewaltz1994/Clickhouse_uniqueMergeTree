#include "getNumberOfPhysicalCPUCores.h"

#include <thread>


unsigned getNumberOfPhysicalCPUCores()
{
    return std::thread::hardware_concurrency();
}

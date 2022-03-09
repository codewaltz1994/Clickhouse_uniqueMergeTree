cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(gpu-acc LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

SET(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

FIND_PACKAGE(CUDA REQUIRED)
# FIND_PACKAGE(cuBLAS REQUIRED)
#Add PThread
FIND_PACKAGE(Threads REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -O3")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CUDA_HOST_COMPILER "gcc")

#include_directories(./lib/rpclib/include)
#include_directories(./lib/nvcomp/include)
#include_directories(./lib/nvcomp/src)
#include_directories(./lib/lz4/lib)
#include_directories(./lib)
#include_directories(./lib/libzmp/include)
#include_directories(./lib/cppzmq)
include_directories(./include)

#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;-pthread;)
#add_subdirectory(lib/rpclib)
#add_subdirectory(lib/nvcomp)
#add_subdirectory(lib/lz4)
#add_subdirectory(lib/libzmq)
#add_subdirectory(lib/cppzmq)
add_subdirectory(src)

set(linklib gpu-acc)# rpc nvcomp lz4 cppzmq)

add_subdirectory(tests)

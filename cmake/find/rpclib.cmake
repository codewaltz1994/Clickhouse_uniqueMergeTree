option(ENABLE_RPCLIB "Enable RPCLIB" ON)

if (NOT ENABLE_RPCLIB)
    return()
endif()

if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/rpclib/CMakeLists.txt")
    message (WARNING "submodule contrib/rpclib is missing. to fix try run: \n git submodule update --init --recursive")
    message (${RECONFIGURE_MESSAGE_LEVEL} "Can't find internal NuRaft library")
	set (USE_RPCLIB 0)
    return()
endif ()

if (NOT OS_FREEBSD)
	set (USE_RPCLIB 1)
	set (RPC_LIBRARY rpc)

	set (RPC_INCLUDE_DIR "${ClickHouse_SOURCE_DIR}/contrib/rpclib/include")

	message (STATUS "Using rpclib=${USE_RPCLIB}: ${RPC_INCLUDE_DIR} : ${RPC_LIBRARY}")
else()
	set (USE_RPCLIB 0)
	message (STATUS "Using internal rpclib library on FreeBSD and Darwin is not supported")
endif()

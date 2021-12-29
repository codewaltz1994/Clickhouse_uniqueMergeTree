include("${ClickHouse_SOURCE_DIR}/cmake/dbms_glob_sources.cmake")
add_headers_and_sources(clickhouse_functions_gpu .)
add_library(clickhouse_functions_gpu ${clickhouse_functions_gpu_sources}
	${clickhouse_functions_gpu_headers})

target_link_libraries(clickhouse_functions_gpu PRIVATE dbms)

if (STRIP_DEBUG_SYMBOLS_FUNCTIONS)
    target_compile_options(clickhouse_functions_gpu PRIVATE "-g0")
endif()

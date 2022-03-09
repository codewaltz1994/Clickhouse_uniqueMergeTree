#aux_source_directory(. DIR_LIB_SRCS)

#add_library (gpu-acc ${DIR_LIB_SRCS})
add_library (gpu-acc UserDefinedFunctions.cu)

target_include_directories(gpu-acc PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu-acc PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(gpu-acc PRIVATE ${CUDA_CUFFT_LIBRARIES})
#target_link_libraries(particles PRIVATE ${CUDA_CUBLAS_LIBRARIES})

# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
target_compile_features(gpu-acc PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(gpu-acc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


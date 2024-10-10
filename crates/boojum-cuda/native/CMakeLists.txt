cmake_minimum_required(VERSION 3.24)
project(boojum_cuda_native)
enable_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()
add_library(boojum_cuda_native STATIC
        ops_cub/common.cuh
        ops_cub/device_radix_sort.cu
        ops_cub/device_reduce.cu
        ops_cub/device_run_length_encode.cu
        ops_cub/device_scan.cu
        poseidon2/queries.cuh
        poseidon2/bn/poseidon2_bn.cuh
        poseidon2/bn/poseidon2_bn_mt.cu
        poseidon2/bn/poseidon2_bn_queries.cu
        poseidon2/bn/poseidon2_bn_st.cu
        poseidon2/gl/poseidon2_gl.cuh
        poseidon2/gl/poseidon2_gl_gates.cuh
        poseidon2/gl/poseidon2_gl_mt.cu
        poseidon2/gl/poseidon2_gl_queries.cu
        poseidon2/gl/poseidon2_gl_st.cu
        poseidon2/gl/poseidon2_gl_st.cuh
        barycentric.cu
        bn254.cu
        bn254.cuh
        blake2s.cu
        carry_chain.cuh
        common.cuh
        context.cu
        context.cuh
        gates.cu
        gates.cuh
        goldilocks.cuh
        goldilocks_extension.cu
        goldilocks_extension.cuh
        memory.cuh
        ntt.cu
        ntt_b2n.cuh
        ntt_n2b.cuh
        ops_complex.cu
        ops_complex.cuh
        ops_simple.cu
        ptx.cuh
)
target_include_directories(boojum_cuda_native PRIVATE ${CMAKE_INSTALL_PREFIX}/generated)
set_target_properties(boojum_cuda_native PROPERTIES CUDA_STANDARD 17)
set_target_properties(boojum_cuda_native PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(boojum_cuda_native PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(boojum_cuda_native PRIVATE --expt-relaxed-constexpr)
target_compile_options(boojum_cuda_native PRIVATE --ptxas-options=-v)
#target_compile_options(boojum_cuda_native PRIVATE -lineinfo)
#target_compile_options(boojum_cuda_native PRIVATE --keep)
install(TARGETS boojum_cuda_native DESTINATION .)

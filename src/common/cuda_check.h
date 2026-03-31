#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(expr)                                                                     \
    do {                                                                                     \
        cudaError_t status__ = (expr);                                                       \
        if (status__ != cudaSuccess) {                                                       \
            std::cerr << "CUDA failure at " << __FILE__ << ":" << __LINE__ << " -> "        \
                      << cudaGetErrorString(status__) << std::endl;                          \
            std::exit(EXIT_FAILURE);                                                         \
        }                                                                                    \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                  \
    do {                                                                                     \
        CUDA_CHECK(cudaGetLastError());                                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                                 \
    } while (0)


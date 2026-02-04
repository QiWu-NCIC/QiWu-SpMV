// spmv_cuda_common.cu
#include "spmv_device_interface.h"  // 你可能需要创建这个头文件
#include <cuda_runtime.h>
#include <cstdio>

// CUDA 内存管理函数
void* allocate_device_memory(size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    void* d_ptr = nullptr;
    cudaMalloc(&d_ptr, bytes);
    return d_ptr;
#else
    return NULL;
#endif
}

void copy_host_to_device(void* dst, const void* src, size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
#endif
}

void copy_device_to_host(void* dst, const void* src, size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);

#endif
}

void memset_device(void* dst, int value, size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    cudaMemset(dst, value, bytes);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

void free_device_memory(void* ptr) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    if (ptr) cudaFree(ptr);
#endif
}
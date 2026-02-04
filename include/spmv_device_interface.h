// spmv_device_interface.h
#ifndef SPMV_DEVICE_INTERFACE_H
#define SPMV_DEVICE_INTERFACE_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// 设备内存管理
void* allocate_device_memory(size_t bytes);
void copy_host_to_device(void* dst, const void* src, size_t bytes);
void copy_device_to_host(void* dst, const void* src, size_t bytes);
void memset_device(void* dst, int value, size_t bytes);
void free_device_memory(void* ptr);

// kernel 启动函数
void launch_spmv_kernel_cuda(void* d_values, void* d_col_idx, void* d_row_ptr,
                             void* d_x, void* d_y, int n, int nnz);

#ifdef __cplusplus
}
#endif

#endif
#include "spmv_benchmark.h"
#include <hip/hip_runtime.h>
#include <string>
using namespace std;

void SpMV_Benchmark::spmv_preprocess_hip() {
    // Preprocessing function - users can add their own preprocessing operations here
    std::cout << "Executing preprocessing steps..." << std::endl;
    
    // Example preprocessing operations:
    // 1. Reorder matrix rows for better cache efficiency
    // 2. Pre-calculate some intermediate results
    // 3. Initialize data structures required for parallel computation
}

__global__ void my_hip_kernel(double* values, int* col_idx, int* row_ptr,
                              double* x, double* y, int n) {
    // HIP kernel 代码
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double sum = 0.0;
        for (int j = row_ptr[idx]; j < row_ptr[idx + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[idx] = sum;
    }
}

void SpMV_Benchmark::spmv_optimized_hip() {
    // Optimized SpMV function for HIP
#ifdef __HIP__  
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(my_hip_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_values, d_col_idx, d_row_ptr, d_x, d_y, n);
#else   
    std::cerr << "HIP not available, Exit." << std::endl;
    exit(1);
#endif
}
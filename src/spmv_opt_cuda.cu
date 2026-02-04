#include "spmv_benchmark.h"
#include <string>
#include <cooperative_groups.h>
using namespace std;

#ifdef __CUDACC__
template <int WARP_SIZE>
__device__ __forceinline__ double warp_reduce_sum(double sum)
{
  for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, i, WARP_SIZE);
  return sum;
}

// CUDA kernel for CSR-Vector multiplication
template <int BLOCK_SIZE, int WARP_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void csr_spmv_kernel(
    const int n,                    // Number of rows in the matrix
    const double* values,          // Values array of the sparse matrix
    const int* col_indices,        // Column indices array
    const int* row_pointers,       // Row pointers array
    const double* x,          // Input dense vector
    double* y                 // Output vector
) {   
    const int lid = threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int nwarp = gridDim.x * BLOCK_SIZE / WARP_SIZE;

    // Loop over rows
    for (int row = gid / WARP_SIZE; row < n; row += nwarp)
    {
        int row_start, row_end;
        row_start = row_pointers[row];
        row_end = row_pointers[row + 1];

        double sum = 0.0;
        if (WARP_SIZE >= 16 && row_end - row_start > 16)
        {
            // ensure aligned memory access to col_indices and values
            int j = row_start - (row_start & (WARP_SIZE - 1)) + lid;

            // accumulate local sums
            if (j >= row_start && j < row_end)
            {
                sum += values[j] * x[col_indices[j]];
            }

            // accumulate local sums
            for (j += WARP_SIZE; j < row_end; j += WARP_SIZE)
            {
                sum += values[j] * x[col_indices[j]];
            }
        }
        else
        {
            // Loop over non-zero elements
            for (int j = row_start + lid; j < row_end; j += WARP_SIZE)
            {
                sum += values[j] * x[col_indices[j]];
            }
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        // First thread of each warp writes result into global memory
        if (lid == WARP_SIZE - 1)
        {
            y[row] += sum;
        }
    }
}
#endif

void SpMV_Benchmark::spmv_preprocess_cuda() {
    // Preprocessing function - users can add their own preprocessing operations here
    std::cout << "Executing preprocessing steps..." << std::endl;
#ifdef __CUDACC__   
    // Example preprocessing operations:
    // 1. Reorder matrix rows for better cache efficiency
    // 2. Pre-calculate some intermediate results
    // 3. Initialize data structures required for parallel computation
#endif
}


void SpMV_Benchmark::spmv_optimized_cuda() {
    // Optimized SpMV function for CUDA
#ifdef __CUDACC__  
    // printf("in cuda kernel!!\n");
    // Configure kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    // Launch kernel
    csr_spmv_kernel<256, 16><<<grid_size, block_size>>>(n, d_values, d_col_idx, d_row_ptr, d_x, d_y);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
#else   
    std::cerr << "CUDA not available, Exit." << std::endl;
    exit(1);
#endif
}
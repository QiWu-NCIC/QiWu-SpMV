#include "spmv_benchmark.h"
#include <string>
using namespace std;

void SpMV_Benchmark::spmv_optimized() {
    // Optimized SpMV function - users can implement their own optimization algorithms here
    // Example: OpenMP parallel version
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}
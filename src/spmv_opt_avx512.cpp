#include "spmv_benchmark.h"
#include <string>
#include <iomanip>
#include <map>
#include <utility>
#include <thread>
#include <mutex>
using namespace std;

void SpMV_Benchmark::spmv_optimized() {
    fill(y.begin(), y.end(), 0.0);
    
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Default to 4 threads if detection fails
    
    vector<thread> threads;
    int rows_per_thread = n / num_threads;
    
    auto spmv_worker = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
#ifdef __AVX512F__
            // Use AVX-512 for optimized computation when possible
            int start = row_ptr[i];
            int end = row_ptr[i + 1];
            int len = end - start;
            
            __m512d acc_vec = _mm512_setzero_pd();
            int j = start;
            
            // Process 8 doubles at a time using AVX-512
            for (; j + 8 <= end; j += 8) {
                __m512d val_vec = _mm512_loadu_pd(&values[j]);
                __m512d x_vec = _mm512_set_pd(
                    x[col_idx[j+7]], x[col_idx[j+6]], x[col_idx[j+5]], x[col_idx[j+4]],
                    x[col_idx[j+3]], x[col_idx[j+2]], x[col_idx[j+1]], x[col_idx[j]]
                );
                __m512d prod = _mm512_mul_pd(val_vec, x_vec);
                acc_vec = _mm512_add_pd(acc_vec, prod);
            }
            
            // Horizontal sum of the accumulator vector
            double temp_result[8];
            _mm512_storeu_pd(temp_result, acc_vec);
            double sum = temp_result[0] + temp_result[1] + temp_result[2] + temp_result[3] +
                        temp_result[4] + temp_result[5] + temp_result[6] + temp_result[7];
            
            // Handle remaining elements
            for (; j < end; ++j) {
                sum += values[j] * x[col_idx[j]];
            }
#else
            // Fallback to regular computation if AVX-512 not available
            double sum = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                sum += values[j] * x[col_idx[j]];
            }
#endif
            y[i] = sum;
        }
    };
    
    // Create and launch threads
    for (int t = 0; t < num_threads; ++t) {
        int start = t * rows_per_thread;
        int end = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        threads.emplace_back(spmv_worker, start, end);
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
}
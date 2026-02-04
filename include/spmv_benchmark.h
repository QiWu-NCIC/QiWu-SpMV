#ifndef SPMV_BENCHMARK_H
#define SPMV_BENCHMARK_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <map>
#include <utility>

class SpMV_Benchmark {
private:
    int n;                      // Matrix dimension
    int nnz_per_row;           // Non-zero elements per row
    std::vector<double> values; // Matrix values
    std::vector<int> col_idx;   // Column indices
    std::vector<int> row_ptr;   // Row pointers
    std::vector<double> x;      // Input vector
    std::vector<double> y;      // Output vector
    std::vector<double> reference_y; // Reference result
    std::string report_filename;
    std::string input_filename;
    std::string kernel_name;

#if CUDA_ENABLED || HIP_ENABLED
    // Device pointers for CUDA
    double *d_values;
    int *d_col_idx;
    int *d_row_ptr;
    double *d_x;
    double *d_y;
#endif

    void generate_report_filename();
    void convert_coo_to_csr(const std::vector<int>& coo_rows, 
                           const std::vector<int>& coo_cols, 
                           const std::vector<double>& coo_vals,
                           int dim, int nnz);

public:
    SpMV_Benchmark(int size, int nnz_pr);
    SpMV_Benchmark(const std::string& mtx_file);

    void load_matrix_from_mtx(const std::string& filename);
    void initialize_matrix();
    void initialize_vectors();
    void allocate_memory();
    void free_memory();
    void run_spmv_kernel();
    void spmv_serial();
    void spmv_preprocess();
    void spmv_preprocess_cuda();
    void spmv_preprocess_hip();
    void spmv_optimized();
    void spmv_optimized_cuda();
    void spmv_optimized_hip();
    std::pair<double, double> benchmark_spmv(int iterations = 5);
    bool validate_correctness();
    double calculate_performance(double spmv_time_us);
    void print_matrix_info();
    void set_kernel_name(const std::string& kernelname);
    void set_report_file(const std::string& reportfile);
    void write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct);
};

#endif // SPMV_BENCHMARK_H
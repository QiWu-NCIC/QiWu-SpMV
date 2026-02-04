#include "spmv_benchmark.h"
#include "spmv_device_interface.h"
#include <string>

using namespace std;

void SpMV_Benchmark::generate_report_filename() {
    std::time_t now = std::time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    std::ostringstream oss;
    oss << "spmv-benchmark-" 
        << std::setfill('0') 
        << std::setw(4) << (timeinfo->tm_year + 1900)
        << std::setw(2) << (timeinfo->tm_mon + 1)
        << std::setw(2) << timeinfo->tm_mday
        << std::setw(2) << timeinfo->tm_hour
        << std::setw(2) << timeinfo->tm_min
        << std::setw(2) << timeinfo->tm_sec
        << ".txt";
    report_filename = oss.str();
}

SpMV_Benchmark::SpMV_Benchmark(int size, int nnz_pr) : n(size), nnz_per_row(nnz_pr) {
    initialize_matrix();
    initialize_vectors();
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    allocate_memory();
#endif
}

SpMV_Benchmark::SpMV_Benchmark(const std::string& mtx_file) {
    // 1. 查找最后一个斜杠（/）的位置
    size_t lastSlashPos = mtx_file.find_last_of('/');
    
    // 2. 处理边界情况
    if (mtx_file.empty()) {  // 空字符串直接返回空
        input_filename = "";
    }
    if (lastSlashPos == string::npos) {  // 没有找到斜杠（直接是文件名）
        input_filename = mtx_file;
    }
    if (lastSlashPos == mtx_file.length() - 1) {  // 斜杠在末尾（无文件名）
        input_filename = "";
    }
    
    // 3. 截取从斜杠下一位到末尾的子串（即文件名+后缀）
    input_filename = mtx_file.substr(lastSlashPos + 1);
    // generate_report_filename();
    load_matrix_from_mtx(mtx_file);
    initialize_vectors();
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    allocate_memory();
#endif
}

void SpMV_Benchmark::load_matrix_from_mtx(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    // Skip comments and read header
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            std::istringstream iss(line);
            iss >> n >> n; // Assuming square matrix
            int total_nnz;
            iss >> total_nnz;
            
            std::vector<int> coo_rows, coo_cols;
            std::vector<double> coo_vals;
            
            // Read COO format data
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                
                std::istringstream iss2(line);
                int row, col;
                double val;
                iss2 >> row >> col >> val;
                
                // Convert to 0-based indexing
                coo_rows.push_back(row - 1);
                coo_cols.push_back(col - 1);
                coo_vals.push_back(val);
            }
            
            // Convert COO to CSR format
            convert_coo_to_csr(coo_rows, coo_cols, coo_vals, n, total_nnz);
            break;
        }
    }
    file.close();
}

void SpMV_Benchmark::convert_coo_to_csr(const std::vector<int>& coo_rows, 
                                       const std::vector<int>& coo_cols, 
                                       const std::vector<double>& coo_vals,
                                       int dim, int nnz) {
    n = dim;
    
    // Initialize CSR structures
    values.resize(nnz);
    col_idx.resize(nnz);
    row_ptr.resize(n + 1, 0);
    
    // Count number of elements in each row
    for (int i = 0; i < nnz; i++) {
        row_ptr[coo_rows[i] + 1]++;
    }
    
    // Compute prefix sum to get row pointers
    for (int i = 1; i <= n; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }
    
    // Fill CSR arrays
    for (int i = 0; i < nnz; i++) {
        int row = coo_rows[i];
        int pos = row_ptr[row];
        
        // Find insertion position for sorted column indices
        int insert_pos = pos;
        while (insert_pos > row_ptr[row] && 
               col_idx[insert_pos - 1] > coo_cols[i]) {
            insert_pos--;
        }
        
        // Shift elements if needed
        for (int k = pos; k > insert_pos; k--) {
            values[k] = values[k - 1];
            col_idx[k] = col_idx[k - 1];
        }
        
        // Insert new element
        values[insert_pos] = coo_vals[i];
        col_idx[insert_pos] = coo_cols[i];
    }
    
    // Recalculate nnz_per_row as average
    nnz_per_row = nnz / n;
}

void SpMV_Benchmark::allocate_memory() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    // 使用封装的接口，而不是直接调用 CUDA API
    d_values = static_cast<double*>(allocate_device_memory(values.size() * sizeof(double)));
    d_col_idx = static_cast<int*>(allocate_device_memory(col_idx.size() * sizeof(int)));
    d_row_ptr = static_cast<int*>(allocate_device_memory(row_ptr.size() * sizeof(int)));
    d_x = static_cast<double*>(allocate_device_memory(x.size() * sizeof(double)));
    d_y = static_cast<double*>(allocate_device_memory(y.size() * sizeof(double)));

    copy_host_to_device(d_values, values.data(), values.size() * sizeof(double));
    copy_host_to_device(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int));
    copy_host_to_device(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int));
    copy_host_to_device(d_x, x.data(), x.size() * sizeof(double));
    memset_device(d_y, 0, y.size() * sizeof(double));
#endif
}

void SpMV_Benchmark::free_memory() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    // Free device memory
    if (d_values) free_device_memory(d_values);
    if (d_col_idx) free_device_memory(d_col_idx);
    if (d_row_ptr) free_device_memory(d_row_ptr);
    if (d_x) free_device_memory(d_x);
    if (d_y) free_device_memory(d_y);
#endif
}

void SpMV_Benchmark::initialize_matrix() {
    // Create sparse matrix - CSR format
    row_ptr.resize(n + 1);
    int total_nnz = n * nnz_per_row;
    values.resize(total_nnz);
    col_idx.resize(total_nnz);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> val_dis(-1.0, 1.0);
    std::uniform_int_distribution<> col_dis(0, n - 1);

    int nnz_count = 0;
    for (int i = 0; i < n; ++i) {
        row_ptr[i] = nnz_count;
        
        // Generate unique column indices for each row
        std::vector<int> temp_cols;
        for (int j = 0; j < nnz_per_row; ++j) {
            int col;
            do {
                col = col_dis(gen);
            } while (std::find(temp_cols.begin(), temp_cols.end(), col) != temp_cols.end());
            
            temp_cols.push_back(col);
            col_idx[nnz_count] = col;
            values[nnz_count] = val_dis(gen);
            nnz_count++;
        }
    }
    row_ptr[n] = nnz_count;
}

void SpMV_Benchmark::initialize_vectors() {
    x.resize(n);
    y.resize(n);
    reference_y.resize(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
        y[i] = 0.0;
        reference_y[i] = 0.0;
    }
}

void SpMV_Benchmark::spmv_serial() {
    // Single-threaded CSR format SpMV calculation (for reference)
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        reference_y[i] = sum;
    }
}

void SpMV_Benchmark::run_spmv_kernel() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    spmv_optimized_cuda();
#elif defined(HIP_ENABLED) && HIP_ENABLED
    spmv_optimized_hip();
#else
    spmv_optimized();
#endif
}

std::pair<double, double> SpMV_Benchmark::benchmark_spmv(int iterations) {
    // Timing preprocessing stage
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    spmv_preprocess();
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count();

    // Timing SpMV execution stage
    auto spmv_start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        run_spmv_kernel();
    }
    
    auto spmv_end = std::chrono::high_resolution_clock::now();
    auto spmv_duration = std::chrono::duration_cast<std::chrono::microseconds>(spmv_end - spmv_start).count();

    // Return preprocessing time and SpMV execution time (average per iteration)
    return std::make_pair(
        static_cast<double>(preprocess_duration),
        static_cast<double>(spmv_duration) / iterations
    );
}

bool SpMV_Benchmark::validate_correctness() {
    // Using single-threaded result as reference to verify current result's correctness
    spmv_serial();  // Calculate reference result
    
    // Re-execute optimized version to get result to be validated
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    memset_device(d_y, 0, y.size() * sizeof(double));
#endif
    run_spmv_kernel();

#if defined(CUDA_ENABLED) && CUDA_ENABLED
    // Copy result back to host
    copy_host_to_device(y.data(), d_y, y.size() * sizeof(double));
    free_memory();
#endif
    // Calculate L2 norm relative error
    double diff_norm = 0.0;  // Square of difference vector L2 norm
    double ref_norm = 0.0;   // Square of reference vector L2 norm
    
    for (int i = 0; i < n; ++i) {
        double diff = y[i] - reference_y[i];
        diff_norm += diff * diff;
        ref_norm += reference_y[i] * reference_y[i];
    }
    
    double relative_error = sqrt(diff_norm) / sqrt(ref_norm);
    
    // Double precision error requirement: relative error should be less than a multiple of machine precision
    const double machine_epsilon = 2.22e-16;
    const double tolerance = 1e6 * machine_epsilon; // Allow million times machine precision error
    
    std::cout << "Reference vector norm: " << sqrt(ref_norm) << std::endl;
    std::cout << "Difference vector norm: " << sqrt(diff_norm) << std::endl;
    std::cout << "Relative error: " << relative_error << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    
    bool passed = relative_error <= tolerance;
    std::cout << "Correctness validation: " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    return passed;
}

double SpMV_Benchmark::calculate_performance(double spmv_time_us) {
    // Calculate performance metric: GFLOPS
    int total_ops = n * nnz_per_row * 2;  // One multiplication and one addition per non-zero element
    double time_seconds = spmv_time_us / 1e6;
    double gflops = (total_ops / time_seconds) / 1e9;
    return gflops;
}

void SpMV_Benchmark::print_matrix_info() {
    int total_nnz = n * nnz_per_row;
    std::cout << "Matrix Name: " << input_filename << std::endl;
    std::cout << "Matrix Size: " << n << "x" << n << std::endl;
    std::cout << "Non-zeros per row: " << nnz_per_row << std::endl;
    std::cout << "Total non-zeros: " << total_nnz << std::endl;
    std::cout << "Sparsity: " << 1.0 - (double)total_nnz / (n * n) << std::endl;
}

void SpMV_Benchmark::set_kernel_name(const std::string& kernelname) {
    kernel_name = kernelname;
}

void SpMV_Benchmark::set_report_file(const std::string& reportfile) {
    if(reportfile == "")
        generate_report_filename();
    report_filename = reportfile;
}

void SpMV_Benchmark::write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct) {
    std::ofstream report_file(report_filename, ios::app);
    if (!report_file.is_open()) {
        std::cerr << "Error: Cannot create report file " << report_filename << std::endl;
        return;
    }

    int total_nnz = n * nnz_per_row;
    
    report_file << "SpMV Benchmark Report\n";
    report_file << "=====================\n";
    report_file << "Date: " << std::asctime(std::localtime(new std::time_t(std::time(nullptr)))) << "\n";
    report_file << "Matrix Information:\n";
    report_file << "  Name: " << input_filename << "\n";
    report_file << "  Size: " << n << "x" << n << "\n";
    report_file << "  Non-zeros per row: " << nnz_per_row << "\n";
    report_file << "  Total non-zeros: " << total_nnz << "\n";
    report_file << "  Sparsity: " << 1.0 - (double)total_nnz / (n * n) << "\n\n";
    
    report_file << "Benchmark Results:\n";
    report_file << "  Kernel: " << kernel_name << "\n";
    report_file << "  Preprocessing time: " << timing_results.first << " microseconds\n";
    report_file << "  Average SpMV execution time: " << timing_results.second << " microseconds\n";
    report_file << "  Performance: " << perf_gflops << " GFLOPS\n\n";
    
    int total_ops = n * nnz_per_row * 2;
    report_file << "Additional Metrics:\n";
    report_file << "  Total operations: " << total_ops << " (multiply-adds)\n";
    report_file << "  Memory accessed (approx): " << 
               (n * sizeof(double) * 2 + n * nnz_per_row * (sizeof(double) + sizeof(int))) << " bytes\n\n";
    
    report_file << "Correctness Validation:\n";
    report_file << "  Result: " << (correct ? "PASSED" : "FAILED") << "\n\n";
    
    report_file.close();
    std::cout << "Report saved to: " << report_filename << std::endl;
}
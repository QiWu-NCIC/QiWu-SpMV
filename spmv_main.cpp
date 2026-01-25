#include "spmv_benchmark.h"
#include <memory>

int main(int argc, char* argv[]) {
    // Configure test parameters
    int matrix_size = 4096;     // Matrix size
    int nnz_per_row = 20;       // Non-zero elements per row

    std::string kernelname = "default_spmv";
    std::string reportfile = "";
    
    std::cout << "SpMV Benchmark Test" << std::endl;   
    std::cout << "===================" << std::endl;

    std::unique_ptr<SpMV_Benchmark> benchmark_ptr;

    if (argc > 3) {
        // 从命令行参数读取.mtx文件
        std::string filename = argv[1];
        kernelname = argv[2];
        reportfile = argv[3];
        std::cout << "Loading matrix from file: " << filename << std::endl;
        benchmark_ptr = std::make_unique<SpMV_Benchmark>(filename);
    } 
    else {
        // 参数不全
        std::cout << "Missing Inputs!! Please  Ensure 2 Inputs (path/to/matrix, YourKernelName). " << std::endl; 
        exit(0);
    }

    // 使用默认的随机矩阵
    // int matrix_size = 4096;
    // int nnz_per_row = 20;
    // reportfile = argv[1];
    // std::cout << "Using randomly generated matrix:" << std::endl;
    // std::cout << "  Size: " << matrix_size << "x" << matrix_size << std::endl;
    // std::cout << "  NNZ per row: " << nnz_per_row << std::endl;
    // benchmark_ptr = std::make_unique<SpMV_Benchmark>(matrix_size, nnz_per_row);
    
    // Create benchmark object
    SpMV_Benchmark& benchmark = *benchmark_ptr;
    benchmark.set_kernel_name(kernelname);
    benchmark.set_report_file(reportfile);
    benchmark.print_matrix_info();
    
    std::cout << "\nRunning SpMV benchmark of "<< kernelname << " with preprocessing..." << std::endl;
    auto [preprocess_time, spmv_avg_time] = benchmark.benchmark_spmv(5);
    double perf_gflops = benchmark.calculate_performance(spmv_avg_time);
    
    std::cout << "Preprocessing time: " << preprocess_time << " microseconds" << std::endl;
    std::cout << "Average SpMV execution time: " << spmv_avg_time << " microseconds" << std::endl;
    std::cout << "Performance: " << perf_gflops << " GFLOPS" << std::endl;
    
    std::cout << "\nValidating correctness..." << std::endl;
    bool correct = benchmark.validate_correctness();
    
    std::cout << "\nAdditional metrics:" << std::endl;
    int total_ops = matrix_size * nnz_per_row * 2;
    std::cout << "Total operations: " << total_ops << " (multiply-adds)" << std::endl;
    std::cout << "Memory accessed (approx): " << (matrix_size * sizeof(double) * 2 + 
               matrix_size * nnz_per_row * (sizeof(double) + sizeof(int))) << " bytes" << std::endl;
    
    // Write detailed report
    benchmark.write_report(std::make_pair(preprocess_time, spmv_avg_time), perf_gflops, correct);
    
    return correct ? 0 : 1;
}
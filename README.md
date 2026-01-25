# SpMV Benchmark Automation Toolkit

A complete toolkit for automating Sparse Matrix-Vector Multiplication (SpMV) benchmarks, processing Matrix Market (.mtx) files, and extracting performance metrics into structured reports.

## ✨ Features

### 🏃 **Automated Benchmark Runner**

* 🔍 Recursively finds all `.mtx` files in directory
* 🕒 Auto-generates timestamp for each run
* ⏱️ Configurable timeouts and parallel execution
* 📝 Comprehensive logging and error handling

### 📊 **Intelligent Report Parser**

* 📈 Extracts 15+ performance metrics
* 🔄 Auto-detects kernel types from reports
* 📁 Smart CSV naming: `{kernel}_spmv_results.csv`
* 📊 Built-in statistical analysis and visualization

## 📋 Software Requirements

### Core Requirements

* **Python** : 3.6 or higher
* **C++ Compiler** : g++ 7+ or clang 5+ (for compiling your SpMV implementation)
* **Memory** : 2GB RAM minimum (more for large matrices)
* **Disk** : Sufficient space for matrix files and results

## 🚀 Quick Start

### 1. Clone and Setup

```
git clone https://github.com/QiWu-NCIC/spmv-benchmark.git
cd spmv-benchmark
mkdir build
cd build
cmake ..
```

### 2. Basic Usage

```
#1. Enter the Root of Benchmark
cd spmv-benchmark

# 2. Run benchmarks on all .mtx files
python auto-spmvbenchmark.py path_to_mtx_folder your_kernel_name

# 3. Parse results to CSV
python data-process.py spmv-benchmark-timestamp
# Output: [kernel_name]_spmv_results.csv (auto-named by kernel)
```

## 📊 CSV Columns Extracted

The parser extracts the following columns from benchmark reports:

| Column                             | Description                   | Unit         | Example                      |
| ---------------------------------- | ----------------------------- | ------------ | ---------------------------- |
| **Basic Matrix Information** |                               |              |                              |
| `matrix_name`                    | Matrix filename               | -            | `dwa512.mtx`               |
| `size_rows`                      | Number of rows                | elements     | `512`                      |
| `size_cols`                      | Number of columns             | elements     | `512`                      |
| `nnz_per_row`                    | Average non-zeros per row     | count        | `4.0`                      |
| `total_nnz`                      | Total non-zero elements       | count        | `2048`                     |
| `sparsity`                       | Matrix sparsity               | ratio (0-1)  | `0.992188`                 |
|                                    |                               |              |                              |
| **Performance Metrics**      |                               |              |                              |
| `kernel`                         | SpMV kernel type              | -            | `default-spmv`             |
| `preprocess_time_us`             | Preprocessing time            | microseconds | `2.0`                      |
| `execution_time_us`              | SpMV execution time           | microseconds | `1738.8`                   |
| `execution_time_ms`              | Execution time (converted)    | milliseconds | `1.7388`                   |
| `performance_gflops`             | Performance in GFLOPS         | GFLOPS       | `0.00235565`               |
|                                    |                               |              |                              |
| **Operation Statistics**     |                               |              |                              |
| `total_operations`               | Total multiply-add operations | count        | `4096`                     |
| `memory_accessed_bytes`          | Approximate memory access     | bytes        | `32768`                    |
|                                    |                               |              |                              |
| **Derived Metrics**          |                               |              |                              |
| `memory_bandwidth_gbs`           | Estimated memory bandwidth    | GB/s         | `0.017`                    |
| `compute_intensity`              | Computational intensity       | FLOPs/Byte   | `0.125`                    |
|                                    |                               |              |                              |
| **Validation & Metadata**    |                               |              |                              |
| `validation_result`              | Correctness check result      | -            | `PASSED` / `FAILED`      |
| `test_date`                      | Timestamp of test execution   | datetime     | `Sat Jan 24 11:23:52 2026` |

### Derived Metrics Calculation

* **Memory Bandwidth** : `memory_accessed_bytes / (execution_time_us / 1e6) / 1e9` GB/s
* **Compute Intensity** : `total_operations / memory_accessed_bytes` FLOPs/Byte
* **Execution Time (ms)** : `execution_time_us / 1000.0`

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository.
2. Implement your code with cpp/CUDA/HIP/..., e.g. the spmv\_opt.cpp
3. Compile the project. See Quick Start.
4. Run the project. See Basic Usage.
5. Open a new branch and push the codes and results to the new branch.

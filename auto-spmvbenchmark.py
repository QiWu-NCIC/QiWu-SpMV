import subprocess
import os
import sys
from datetime import datetime

def run_cpp_benchmark(folder_path, user_string):
    # 1. check path valid
    if not os.path.isdir(folder_path):
        print(f"Error: path '{folder_path}' invalid")
        return

    # 2. get all .mtx 
    mtx_files = [f for f in os.listdir(folder_path) if f.endswith('.mtx')]
    
    if not mtx_files:
        print("Not Found .mtx file")
        return

    # 3. run benchmark
    cpp_executable = "./build/spmvBenchmark" 
    # system time in second
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    report_name = "spmv-test-"+current_time

    for mtx in mtx_files:
        mtx_path = os.path.join(folder_path, mtx)     
        # exe: ./executable [mtx_path] [user_string] [current_time]
        command = [cpp_executable, mtx_path, user_string, report_name]
        
        print(f"Processing: {mtx} ...")
        
        try:
            # exe and wait
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            
        except subprocess.CalledProcessError as e:
            print(f"File {mtx} ERROR: {e}")
            # print(e.stderr)

if __name__ == "__main__":
    # parameters
    # python auto-spmvbenchmark.py [folder_path] [user_string]
    if len(sys.argv) < 3:
        print("Usage: python auto-spmvbenchmark.py <path/to/your/mtx> <your/kernel/name>")
    else:
        path_arg = sys.argv[1]
        string_arg = sys.argv[2]
        run_cpp_benchmark(path_arg, string_arg)
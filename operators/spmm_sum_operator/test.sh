rm -rf build
mkdir build
cd build
cmake .. -DBUILD_PYBIND11=ON -DRUN_MODE=npu
make -j
cd ..
cd spmm_sum_benchmark
export LD_LIBRARY_PATH=../build/lib:$LD_LIBRARY_PATH
python3 compare_cpu_npu.py
python3 dgl_benchmark_npu.py
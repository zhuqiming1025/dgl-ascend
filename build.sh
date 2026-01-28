bash script/build_dgl_ascend.sh -c -s Ascend910B4

# Copy AscendC kernel library to build directory for runtime
if [ -f "build/lib/libdgl_ascendc_kernels.so" ] && [ ! -f "build/libdgl_ascendc_kernels.so" ]; then
    cp build/lib/libdgl_ascendc_kernels.so build/
    echo "Copied libdgl_ascendc_kernels.so to build directory"
fi

# Set LD_LIBRARY_PATH to include build directory so libdgl.so can find libdgl_ascendc_kernels.so
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/zqm1/dgl-ascend/build:$LD_LIBRARY_PATH

cd python
pip install -e .
cd ..
# python3 tests/ascend/test_spmm_npu.py
python3 tests/ascend/test_spmm_correctness.py
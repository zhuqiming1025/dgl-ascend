cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# ${KERNEL_FILES} are used to compile library, push files written by ascendc in ${KERNEL_FILES}.
# ref to cmake/npu.cmake ascendc_library, cmake/cpu.cmake add_library
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/spmm_sum.cpp)

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()
add_executable(ascendc_kernels_bbit ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
)

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Pybind11 support
option(BUILD_PYBIND11 "Build pybind11 extension" OFF)
if(BUILD_PYBIND11)
    add_library(pybind11_lib SHARED pybind11.cpp)
    target_link_libraries(pybind11_lib PRIVATE
        ascendc_kernels_${RUN_MODE}
        torch_npu
    )
    
    execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TORCH_PATH
    )
    message("TORCH_PATH is ${TORCH_PATH}")
    set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
    execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TORCH_NPU_PATH
    )
    message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")
    
    target_link_directories(pybind11_lib PRIVATE
        ${TORCH_PATH}/lib
        ${TORCH_NPU_PATH}/lib
    )
    
    target_include_directories(pybind11_lib PRIVATE
        ${TORCH_NPU_PATH}/include
        ${TORCH_PATH}/include
        ${TORCH_PATH}/include/torch/csrc/api/include
    )
    
    execute_process(COMMAND python3 -m pybind11 --includes
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYBIND11_INC
    )
    string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
    target_compile_options(pybind11_lib PRIVATE
        ${PYBIND11_INC}
        -D_GLIBCXX_USE_CXX11_ABI=0
    )
    
    execute_process(COMMAND python3-config --extension-suffix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYBIND11_SUFFIX
    )
    set_target_properties(pybind11_lib PROPERTIES
        OUTPUT_NAME spmm_sum${PYBIND11_SUFFIX}
        PREFIX "" SUFFIX ""
    )
endif()

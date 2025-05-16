cmake_minimum_required(VERSION 3.10)
project(SpMVParallel C CUDA)

set(CMAKE_C_STANDARD 17)

# Opzioni per selezionare CUDA o OpenMP
option(ENABLE_CUDA "Compile CUDA version" OFF)
option(ENABLE_OPENMP "Compile OpenMP version" ON)

# Determina se stiamo compilando sul server di dipartimento
cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
message(STATUS "Building on host: ${HOSTNAME}")

# Percorsi degli header
include_directories(${CMAKE_SOURCE_DIR}/libs ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/cuda_libs)

# --------------------------------------
# OPENMP CONFIGURATION
# --------------------------------------
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

    set(OPENMP_SOURCES
            main.c
            libs/mmio.c
            src/csr_matrix.c
            src/matrix_parser.c
            src/utility.c
            src/hll_matrix.c
            libs/hll_matrix.h
            src/performance_calculate.c
            libs/performance_calculate.h
    )

    add_executable(SpMV_OpenMP ${OPENMP_SOURCES})
    target_link_libraries(SpMV_OpenMP PRIVATE OpenMP::OpenMP_C m)

    message(STATUS "OpenMP version enabled")
endif()

# --------------------------------------
# CUDA CONFIGURATION
# --------------------------------------
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native -lineinfo -DCUDA_COMPILE")

    set(CUDA_SOURCES
            main_cuda.cu
            cuda_src/hll_matrix.cu
            cuda_src/csr_matrix_cuda.cu
            cuda_src/matrix_parser.cu
            cuda_src/performance_calculate.cu
            cuda_src/utility.cu
    )

    add_executable(SpMV_CUDA ${CUDA_SOURCES})
    target_include_directories(SpMV_CUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_libs)

    set_target_properties(SpMV_CUDA
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )

    add_library(mmio STATIC libs/mmio.c)
    target_link_libraries(SpMV_CUDA PRIVATE mmio CUDA::cudart m)

    message(STATUS "CUDA version enabled")
endif()

# --------------------------------------
# OUTPUT SUMMARY
# --------------------------------------
if(NOT ENABLE_CUDA AND NOT ENABLE_OPENMP)
    message(FATAL_ERROR "You must enable at least one of ENABLE_CUDA or ENABLE_OPENMP")
endif()

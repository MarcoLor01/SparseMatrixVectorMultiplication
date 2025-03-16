cmake_minimum_required(VERSION 3.10)
project(SpMVParallel C)

set(CMAKE_C_STANDARD 17)

# Trova OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3 -march=native")

# Determina se stiamo compilando sul server di dipartimento
cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
message(STATUS "Building on host: ${HOSTNAME}")

# Definisci tutti i file sorgente
set(SOURCE_FILES
        main.c
        libs/mmio.c
        src/csr_matrix.c
        src/matrix_parser.c
        src/utility.c
        src/hll_matrix.c
        libs/hll_matrix.h
        src/performance_calculate.c
        src/performance_calculate.c
        libs/performance_calculate.h
        libs/pre_matrix.h
)

# Crea l'eseguibile direttamente con tutti i file sorgente
add_executable(SpMVParallel ${SOURCE_FILES})

# Collega OpenMP all'eseguibile
target_link_libraries(SpMVParallel PRIVATE
        OpenMP::OpenMP_C
        m
)

# Specifica dove trovare gli header file
target_include_directories(SpMVParallel PRIVATE
        ${CMAKE_SOURCE_DIR}/libs
        ${CMAKE_SOURCE_DIR}/src
        ${ADDITIONAL_INCLUDE_DIRS}  # Aggiunge percorsi aggiuntivi in base all'ambiente
)

# Aggiungi directory per le librerie specifiche del server (se presenti)
if(ADDITIONAL_LIB_DIRS)
    target_link_directories(SpMVParallel PRIVATE ${ADDITIONAL_LIB_DIRS})
endif()
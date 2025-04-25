#ifndef CSR_MATRIX_CUDA_CUH
#define CSR_MATRIX_CUDA_CUH

#include <cuda_runtime.h>
#include "matrix_parser.cuh"
#include <utility.cuh>

typedef struct {
    int M;              // Numero di righe
    int N;              // Numero di colonne
    int nz;             // Elementi non zero
    int *row_ptr;       // Array dei puntatori alle righe
    int *col_idx;       // Array degli indici delle colonne
    double *values;     // Array dei valori
    MM_typecode type;   // Tipo della matrice
} CSRMatrix;

void init_csr_matrix(CSRMatrix *mat);
void free_csr_matrix(CSRMatrix *mat);
int convert_in_csr(const PreMatrix *pre, CSRMatrix *csr);
void csr_matrix_vector_mult(const int num_row,
    const int *row_ptr, const int *col_idx, const double *values,
    const double *x, double *y);
void print_csr_matrix(const CSRMatrix *mat);
// Kernel CUDA per SpMV - Naive
__global__ void spmv_csr_naive_kernel(
    int M,
    const int *row_ptr,
    const int *col_idx,
    const double *values,
    const double *x,
    double *y);
__global__ void spmv_csr_warp_kernel(
    int M,              // Numero di righe
    const int *row_ptr, // Array dei puntatori alle righe
    const int *col_idx, // Array degli indici delle colonne
    const double *values, // Array dei valori
    const double *x,    // Vettore di input
    double *y           // Vettore di output
);
__global__ void spmv_csr_warp_shared_memory_kernel(
    int M,
    const int *row_ptr,
    const int *col_idx,
    const double *values,
    const double *x,
    double *y
);
#endif // CSR_MATRIX_CUDA_CUH

#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <mmio.h>
#include "pre_matrix.h"

// Struttura per matrice CSR
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
void print_csr_matrix(const CSRMatrix *mat);
void csr_matrix_vector_mult(int num_row, const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y);
void spvm_csr_parallel(const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y, int num_threads, const int *thread_row_start, const int *thread_row_end);
void spvm_parallel_dynamic(int num_row, const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y, int num_threads);
int prepare_thread_distribution(int num_row, const int *row_ptr, int num_threads, long long total_nnz, int *thread_row_start, int *thread_row_end);
void spvm_csr_parallel_unrolling(const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y, int num_threads, const int *thread_row_start, const int *thread_row_end);
#endif
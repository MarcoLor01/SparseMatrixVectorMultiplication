//
// Created by HP on 08/03/2025.
//

#ifndef ELL_MATRIX_H
#define ELL_MATRIX_H
#include "pre_matrix.h"
#include <stddef.h>
#include <mmio.h>

#define HACK_SIZE 32

// Struttura blocco ELLPACK con layout trasposto
typedef struct {
    int M;              // Numero di righe
    int N;              // Numero di colonne
    int MAXNZ;          // Max nonzeri per riga
    int *JA;            // Array di indici di colonna (MAXNZ x M) linearizzato
    double *AS;         // Array di coefficienti (MAXNZ x M) linearizzato
} ELLPACKBlock;

// Struttura HLL modificata (come lista di blocchi)
typedef struct {
    int num_blocks;     // Numero di blocchi
    ELLPACKBlock *blocks; // Array di blocchi ELLPACK
} HLLMatrix;

void init_hll_matrix(HLLMatrix *hll);
int convert_in_hll(const PreMatrix *pre, HLLMatrix *hll);
void free_hll_matrix(HLLMatrix *hll);
void spmv_hll(const HLLMatrix *A, const double *x, double *y, int num_threads,
              int const *thread_block_start, int const *thread_block_end);
int prepare_thread_distribution_hll(const HLLMatrix *matrix, int num_threads,
                                    int *thread_block_start, int *thread_block_end);
void printHLLMatrix(HLLMatrix *hll);
void spmv_hll_simd(HLLMatrix *A, const double *x, double *y, int num_threads,
                   int const *thread_block_start, int const *thread_block_end);
void spmv_hll_serial(const HLLMatrix *A, const double *x, double *y);
#endif //ELL_MATRIX_H

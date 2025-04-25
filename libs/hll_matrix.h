//
// Created by HP on 08/03/2025.
//

#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#include <stddef.h>
#include <mmio.h>
#include <matrix_parser.h>

#define HACK_SIZE 32

// Struttura blocco ELLPACK con layout trasposto
typedef struct {
    int M;              // Numero di righe
    int N;              // Numero di colonne
    int MAXNZ;          // Max nonzeri per riga
    int *JA;            // Array di indici di colonna (MAXNZ x M) linearizzato
    double *AS;         // Array di coefficienti (MAXNZ x M) linearizzato
} ELLPACKBlock;

// Struttura HLL
typedef struct {
    int num_blocks;     // Numero di blocchi
    ELLPACKBlock *blocks; // Array di blocchi ELLPACK
} HLLMatrix;

void init_hll_matrix(HLLMatrix *hll);
int convert_to_hll(const PreMatrix *pre, HLLMatrix *hll);
void free_hll_matrix(HLLMatrix *hll);
void printHLLMatrix(HLLMatrix *hll);
void spmv_hll_serial(int num_blocks, const ELLPACKBlock *blocks, const double *x, double *y);
int prepare_thread_distribution_hll(const HLLMatrix *matrix, int num_threads,
                                    int **thread_block_start, int **thread_block_end);
void spmv_hll(const ELLPACKBlock *blocks, const double *x, double *y, int num_threads,
              int const *thread_block_start, int const *thread_block_end);
void spmv_hll_simd(const ELLPACKBlock *blocks, const double *x, double *y, int num_threads,
                   int const *thread_block_start, int const *thread_block_end);
int save_hll_memory_stats(const HLLMatrix *hll, const char *matrix_name, const char *filename);
#endif //HLL_MATRIX_H

#ifndef HLL_MATRIX_CUH
#define HLL_MATRIX_CUH

#include <stddef.h>
#include <mmio.h>
#include <matrix_parser.cuh>
#include <utility.cuh>

#define HACK_SIZE 32
#define WARP_SIZE 32

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

// Struttura per ordinare gli elementi per riga e colonna
typedef struct {
    int col;
    double val;
} ColValPair;


int compareColValPair(const void *a, const void *b);
int compute_max_MAXNZ(const ELLPACKBlock* blocks, int num_blocks);
void init_hll_matrix(HLLMatrix *hll);
int convert_to_hll(const PreMatrix *pre, HLLMatrix *hll);
void free_hll_matrix(HLLMatrix *hll);
void printHLLMatrix(HLLMatrix *hll);
void spmv_hll_serial(int num_blocks, const ELLPACKBlock *blocks, const double *x, double *y);
__global__ void spmv_hll_naive_kernel(int total_M, const ELLPACKBlock *blocks,
    const double *x, double *y);
__global__ void spmv_hll_warp_shared_kernel_v1(
    int total_M,
    const ELLPACKBlock *blocks,
    const double *x,
    double *y);
__global__ void spmv_hll_warp_kernel(
    int total_M,
    const ELLPACKBlock *blocks, // Puntatore all'array di blocchi su GPU
    const double *x,
    double *y);
#endif //HLL_MATRIX_CUH

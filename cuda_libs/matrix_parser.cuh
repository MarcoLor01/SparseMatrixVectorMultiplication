#ifndef PRE_MATRIX_CUH
#define PRE_MATRIX_CUH
#include <mmio.h>
#include <stdbool.h>

typedef struct {
    int M;              // Numero di righe
    int N;              // Numero di colonne
    int nz;             // Elementi non zero
    int *I;             // Array degli indici di riga
    int *J;             // Array degli indici di colonna
    double *val;        // Array dei valori
    MM_typecode type;   // Tipo della matrice
} PreMatrix;

void init_pre_matrix(PreMatrix *mat);
void free_pre_matrix(PreMatrix *mat);
int read_matrix_market(const char *filename, PreMatrix *mat);
void print_pre_matrix(PreMatrix *mat, const bool full_print);

#endif
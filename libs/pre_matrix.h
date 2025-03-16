#ifndef MATRIX_TYPES_H
#define MATRIX_TYPES_H

#include "mmio.h"
// Struttura per matrice iniziale

typedef struct {
    int M;              // Numero di righe
    int N;              // Numero di colonne
    int nz;             // Elementi non zero
    int *I;             // Array degli indici di riga
    int *J;             // Array degli indici di colonna
    double *val;        // Array dei valori
    MM_typecode type;   // Tipo della matrice
} PreMatrix;

#endif

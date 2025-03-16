#ifndef PRE_MATRIX_H
#define PRE_MATRIX_H
#include <stdbool.h>

#include "pre_matrix.h"


void init_pre_matrix(PreMatrix *mat);
void free_pre_matrix(PreMatrix *mat);
int read_matrix_market(const char *filename, PreMatrix *mat);
void print_pre_matrix(const PreMatrix *mat, bool full_print);

#endif

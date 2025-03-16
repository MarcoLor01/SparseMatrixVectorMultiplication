#ifndef UTILITY_H
#define UTILITY_H
#include <stdbool.h>

#include "hll_matrix.h"

int get_max_threads();
double checkDifferences(const double *y_h, const double *y_SerialResult, int matrix_row, bool full_print);
double calculate_flops(int nz, double time);
void write_results_to_csv(const char *matrix_name, int num_rows, int num_cols, int nz,
                          int num_threads, double time_serial, double time_serial_hll, double time_parallel,
                          double time_parallel_unrolling, double time_parallel_hll, double time_parallel_hll_simd, double speedup_parallel,
                          double speedup_unrolling, double speedup_hll, double speedup_hll_simd, double efficiency_parallel,
                          double efficiency_unrolling, double efficiency_hll, double efficiency_hll_simd, double flops_serial, double avg_flops_hll_serial,
                          double flops_parallel, double flops_parallel_unrolling, double flops_parallel_hll, double flops_parallel_hll_simd, const char *output_file);
void calculate_and_print_thread_averages(int num_thread, const char *output_summary_file);
void swap(int *a, int *b);
void swap_double(double *a, double *b);
size_t partition(int *col_idx, double *values, size_t low, size_t high);
void sort_row(int *col_idx, double *values, size_t low, size_t high);
#endif //UTILITY_H
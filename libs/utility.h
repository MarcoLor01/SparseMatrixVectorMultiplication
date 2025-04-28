#ifndef UTILITY_H
#define UTILITY_H
#include <stdbool.h>
#include <matrix_parser.h>
#include <performance_calculate.h>

#define ITERATION_SKIP 5
#define FREE_CHECK(ptr)           \
do {                              \
    if ((ptr) != NULL) {          \
        free(ptr);                \
        (ptr) = NULL;             \
    }                             \
} while (0)

void init_vector_at_one(double *v, const int size);
double checkDifferences(const double *y_h, const double *y_SerialResult, int matrix_row, bool full_print);
void write_results_to_csv(const char *matrix_name, const int num_rows, const int num_cols, const int nz,
                          const int num_threads, const double time_serial, const double time_serial_hll, const double time_parallel,
                          const double time_parallel_simd, const double time_parallel_hll, const double time_parallel_hll_simd,
                          DiffMetrics error_csr, DiffMetrics error_hll, DiffMetrics error_csr_simd, DiffMetrics error_hll_simd, const double speedup_parallel,
                          const double speedup_simd, const double speedup_hll, const double speedup_hll_simd, const double efficiency_parallel,
                          const double efficiency_simd, const double efficiency_hll, const double efficiency_hll_simd, const double flops_serial, const double avg_flops_hll_serial,
                          const double flops_parallel, const double flops_parallel_simd, const double flops_parallel_hll, const double flops_parallel_hll_simd, const char *output_file);
void calculate_and_print_thread_averages(int num_thread, const char *output_summary_file);
void swap(int *a, int *b);
void swap_double(double *a, double *b);
size_t partition(int *col_idx, double *values, size_t low, size_t high);
void clear_cache(size_t clear_size_mb);
void sort_row(int *col_idx, double *values, size_t low, size_t high);
void create_directory(const char *path);
int process_matrix_file(const char *filepath, PreMatrix *pre_mat);
#endif //UTILITY_H
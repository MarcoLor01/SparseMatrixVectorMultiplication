#ifndef UTILITY_CUH
#define UTILITY_CUH
#define ITERATION_SKIP 5
#define MAX_CACHE 1024
#define WARP_SIZE 32
#include <cuda_runtime.h>
#include <matrix_parser.cuh>
#include <performance_calculate.cuh>

#define FREE_CHECK(ptr)           \
do {                              \
    if ((ptr) != nullptr) {       \
        free(ptr);                \
        (ptr) = nullptr;          \
    }                             \
} while (0)

void init_vector_at_one(double *v, const int size);

void write_results_to_csv(const char *matrix_name, const int num_rows, const int num_cols, const int nz,
                          const double time_serial, const double time_serial_hll, const double time_row_csr,
                          const double time_warp_csr, const double time_warp_csr_shared, const double time_warp_shared_hll,
                          const double time_row_hll, const double time_warp_hll,
                          const double flops_serial, const double avg_flops_hll_serial, const double flops_row_csr,
                          const double flops_warp_csr, const double flops_row_hll, const double flops_warp_hll, const double flop_warp_csr_shared,
                          const double flops_warp_shared_hll, DiffMetrics mediumCsrParallel, DiffMetrics mediumCsrWarp,
                          DiffMetrics mediumCsrWarpShared, DiffMetrics mediumHllNaive, DiffMetrics mediumHllWarp, DiffMetrics mediumHllWarpShared,
                          const char *output_file);

void calculate_and_print_thread_averages(int num_thread, const char *output_summary_file);
void swap(int *a, int *b);
void swap_double(double *a, double *b);
size_t partition(int *col_idx, double *values, size_t low, size_t high);
void sort_row(int *col_idx, double *values, size_t low, size_t high);
void clear_gpu_cache(size_t clear_size_mb);
int process_matrix_file(const char *filepath, PreMatrix *pre_mat);
void create_directory(const char *path);
int clear_directory(const char *path);
#endif //UTILITY_CUH
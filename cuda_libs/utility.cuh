#ifndef UTILITY_CUH
#define UTILITY_CUH
#include <stdbool.h>

#define MAX_CACHE 1024
#define WARP_SIZE 32

#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr,                                                      \
                "Errore CUDA in %s:%d – \"%s\"\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

#define FREE_CHECK(ptr)           \
do {                              \
    if ((ptr) != nullptr) {       \
        free(ptr);                \
        (ptr) = nullptr;          \
    }                             \
} while (0)

void init_vector_at_one(double *v, const int size);

bool checkDifferences(
    const double* ref,
    const double* res,
    int n,
    double abs_tol = 1e-9, // Tolleranza assoluta di default (più stringente)
    double rel_tol = 1e-7, // Tolleranza relativa di default (più stringente)
    bool print_summary = true,
    bool print_first_failure = true);

void write_results_to_csv(const char *matrix_name, const int num_rows, const int num_cols, const int nz,
                          const double time_serial, const double time_serial_hll, const double time_row_csr,
                          const double time_warp_csr, const double time_warp_csr_shared, const double time_warp_shared_hll,
                          const double time_row_hll, const double time_warp_hll,
                          const double flops_serial, const double avg_flops_hll_serial, const double flops_row_csr,
                          const double flops_warp_csr, const double flops_row_hll, const double flops_warp_hll, const double flop_warp_csr_shared,
                          const double flops_warp_shared_hll,
                          const char *output_file);
void calculate_and_print_thread_averages(int num_thread, const char *output_summary_file);
void swap(int *a, int *b);
void swap_double(double *a, double *b);
size_t partition(int *col_idx, double *values, size_t low, size_t high);
void sort_row(int *col_idx, double *values, size_t low, size_t high);
void clear_gpu_cache(size_t clear_size_mb);
#endif //UTILITY_CUH
//
// Created by HP on 04/03/2025.
//
#include "utility.h"
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <mmio.h>

#define MAX_CACHE 1024
// Inizializza vettori a 1
void init_vector_at_one(double *v, const int size) {
    for (int i = 0; i < size; i++) {
        v[i] = 1.0;
    }
}

// Funzioni di utilità per l'ordinamento
void swap(int *a, int *b) {
    const int temp = *a;
    *a = *b;
    *b = temp;
}


void swap_double(double *a, double *b) {
    const double temp = *a;
    *a = *b;
    *b = temp;
}

// Partizione per quicksort
size_t partition(int *col_idx, double *values, const size_t low, const size_t high) {
    const int pivot = col_idx[high];
    size_t i = low - 1;

    for (size_t j = low; j < high; j++) {
        if (col_idx[j] <= pivot) {
            i++;
            swap(&col_idx[i], &col_idx[j]);
            swap_double(&values[i], &values[j]);
        }
    }

    swap(&col_idx[i + 1], &col_idx[high]);
    swap_double(&values[i + 1], &values[high]);
    return i + 1;
}



// Quicksort iterativo
void sort_row(int *col_idx, double *values, size_t low, size_t high) {
    if (low < high) {
        if (high - low > 10000) {
            // Versione iterativa per array grandi
            size_t stack[64];
            int top = -1;

            stack[++top] = low;
            stack[++top] = high;

            while (top >= 0) {
                high = stack[top--];
                low = stack[top--];

                size_t p = partition(col_idx, values, low, high);

                if (p > low) {
                    stack[++top] = low;
                    stack[++top] = p - 1;
                }

                if (p < high) {
                    stack[++top] = p + 1;
                    stack[++top] = high;
                }
            }
        } else {
            // Versione ricorsiva per array più piccoli
            const size_t pi = partition(col_idx, values, low, high);
            if (pi > 0) sort_row(col_idx, values, low, pi - 1);
            sort_row(col_idx, values, pi + 1, high);
        }
    }
}

void write_results_to_csv(const char *matrix_name, const int num_rows, const int num_cols, const int nz,
                          const double time_serial, const double time_serial_hll, const double time_row_csr,
                          const double time_warp_csr, const double time_warp_csr_shared, const double time_warp_shared_hll,
                          const double time_row_hll, const double time_warp_hll,
                          const double flops_serial, const double avg_flops_hll_serial, const double flops_row_csr,
                          const double flops_warp_csr, const double flops_row_hll, const double flops_warp_hll, const double flops_warp_csr_shared,
                          const double flops_warp_shared_hll, DiffMetrics mediumCsrParallel, DiffMetrics mediumCsrWarp,
                          DiffMetrics mediumCsrWarpShared, DiffMetrics mediumHllNaive,
                          DiffMetrics mediumHllWarp, DiffMetrics mediumHllWarpShared,
                          const char *output_file) {
    FILE *fp = fopen(output_file, "a+");  // Append + read

    if (fp == NULL) {
        printf("Errore nell'apertura del file %s\n", output_file);
    return;
    }

    // Verifica se il file è vuoto per decidere se scrivere l'intestazione
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    if (size == 0) {
        fprintf(fp, "matrix_name,rows,cols,nonzeros,"
            "time_serial,time_serial_hll,time_row_csr,time_warp_csr,time_warp_shared_csr,time_row_hll,time_warp_hll,time_warp_shared_hll,"
            "flops_serial,avg_flops_hll_serial,flops_row_csr,flops_warp_csr,flops_warp_csr_shared,flops_row_hll,flops_warp_hll,flops_warp_shared_hll,"
            "relative_error_row_csr,absolute_error_row_csr,"
            "relative_error_warp_csr,absolute_error_warp_csr,"
            "relative_error_warp_shared_csr,absolute_error_warp_shared_csr,"
            "relative_error_row_hll,absolute_error_row_hll,"
            "relative_error_warp_hll,absolute_error_warp_hll,"
            "relative_error_warp_shared_hll,absolute_error_warp_shared_hll\n");
    }

    // Scrivi una riga di dati
    fprintf(fp, "%s,%d,%d,%d,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
            matrix_name, num_rows, num_cols, nz,
            time_serial, time_serial_hll, time_row_csr, time_warp_csr, time_warp_csr_shared, time_row_hll, time_warp_hll, time_warp_shared_hll,
            flops_serial, avg_flops_hll_serial, flops_row_csr, flops_warp_csr, flops_warp_csr_shared, flops_row_hll, flops_warp_hll, flops_warp_shared_hll,
            mediumCsrParallel.mean_rel_err, mediumCsrParallel.mean_abs_err, mediumCsrWarp.mean_rel_err, mediumCsrWarp.mean_abs_err,
            mediumCsrWarpShared.mean_rel_err, mediumCsrWarpShared.mean_abs_err, mediumHllNaive.mean_rel_err, mediumHllNaive.mean_abs_err,
            mediumHllWarp.mean_rel_err, mediumHllWarp.mean_abs_err, mediumHllWarpShared.mean_rel_err, mediumHllWarpShared.mean_abs_err);
    fclose(fp);

}


// Kernel per scrivere nella memoria e svuotare la cache
__global__ void clear_cache_kernel(char *buffer, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] = (char)(idx % 256);
    }
}

// Funzione per svuotare la cache GPU
void clear_gpu_cache(size_t clear_size_mb) {
    size_t bytes_to_clear = clear_size_mb * 1024 * 1024;
    char *d_dummy_buffer = NULL;

    printf("Clearing GPU cache with %zu MB buffer...\n", clear_size_mb);

    // Alloca memoria sulla GPU
    cudaError_t err = cudaMalloc((void**)&d_dummy_buffer, bytes_to_clear);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU cache clearing buffer: %s\n", cudaGetErrorString(err));
        return;
    }

    // Definisci dimensioni per il kernel di pulizia
    int threadsPerBlock = 256;
    int blocksPerGrid = (bytes_to_clear + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel per scrivere sulla memoria allocata
    clear_cache_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dummy_buffer, bytes_to_clear);

    // Sincronizzazione per assicurarsi che il kernel sia completato
    cudaDeviceSynchronize();

    // Libera la memoria
    cudaFree(d_dummy_buffer);

    printf("GPU cache clearing attempt finished.\n");
}

int process_matrix_file(const char *filepath, PreMatrix *pre_mat) {
    init_pre_matrix(pre_mat);

    printf("\n===========================================\n");
    printf("Elaborazione matrice: %s\n", filepath);
    printf("===========================================\n");

    if (read_matrix_market(filepath, pre_mat) != 0) {
        printf("Errore nella lettura della matrice\n");
        return -1;
    }
    return 0;
}

int clear_directory(const char *path) {
    DIR *dir = opendir(path);
    if (dir == NULL) {
        perror("Errore nell'aprire la directory");
        return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // Evita di rimuovere "." e ".."
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", path, entry->d_name);

            if (remove(file_path) == -1) {
                perror("Errore nell'eliminare il file");
                closedir(dir);
                return -1;
            }
        }
    }

    closedir(dir);
    return 0;
}

// Funzione per svuotare il contenuto di una directory
void create_directory(const char *path) {
    if (mkdir(path, 0777) == -1) {
        if (errno == EEXIST) {
            printf("La directory 'result' esiste già. Svuotiamo il suo contenuto...\n");
            if (clear_directory(path) == -1) {
                printf("Errore nel svuotare la directory.\n");
                exit(EXIT_FAILURE);
            }
            printf("La directory è stata svuotata.\n");
        } else {
            perror("Errore nella creazione della directory");
            exit(EXIT_FAILURE);
        }
    } else {
        printf("Directory 'result' creata con successo.\n");
    }
}




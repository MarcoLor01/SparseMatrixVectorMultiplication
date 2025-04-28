//
// Created by HP on 04/03/2025.
//

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include "utility.h"
#include <stdlib.h>     // per exit, EXIT_FAILURE
#include <string.h>     // per strcmp
#include <dirent.h>     // per opendir, readdir, closedir, DIR, struct dirent
#include <sys/stat.h>   // per mkdir
#include <errno.h>      // per errno, EEXIST
#include "matrix_parser.h"

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


double checkDifferences(const double *y_h, const double *y_SerialResult, const int matrix_row, bool full_print) {
    double totalRelativeDiff = 0.0;  // Somma delle differenze relative
    int count = 0;  // Conta gli errori significativi

    for (int i = 0; i < matrix_row; i++) {
        const double toleranceRel = 1e-6;
        const double absTolerance = 1e-7;
        const double currentDiff = fabs(y_SerialResult[i] - y_h[i]);  // Differenza assoluta
        double maxAbs = fmax(fabs(y_SerialResult[i]), fabs(y_h[i]));  // Max tra i valori assoluti

        // Se il massimo valore è piccolo, consideriamo una tolleranza relativa
        if (maxAbs < toleranceRel) {
            maxAbs = toleranceRel;
        }

        const double relativeDiff = currentDiff > absTolerance ? currentDiff / maxAbs : 0.0;  // Calcola differenza relativa

        // Se la differenza è significativa, accumula il valore
        if (relativeDiff > toleranceRel) {
            if (full_print == true) {
                printf("Errore: y[%d] calcolato (%.10f) differisce dal valore seriale (%.10f).\n", i, y_h[i], y_SerialResult[i]);
            }
            totalRelativeDiff += relativeDiff;
            count++;
        }
    }

    // Restituisce la media delle differenze relative se sono stati trovati errori significativi
    return count > 0 ? totalRelativeDiff / count : 0.0;
}

void write_results_to_csv(const char *matrix_name, const int num_rows, const int num_cols, const int nz,
                          const int num_threads, const double time_serial, const double time_serial_hll, const double time_parallel,
                          const double time_parallel_simd, const double time_parallel_hll, const double time_parallel_hll_simd,
                          DiffMetrics error_csr, DiffMetrics error_hll, DiffMetrics error_csr_simd, DiffMetrics error_hll_simd, const double speedup_parallel,
                          const double speedup_simd, const double speedup_hll, const double speedup_hll_simd, const double efficiency_parallel,
                          const double efficiency_simd, const double efficiency_hll, const double efficiency_hll_simd, const double flops_serial, const double avg_flops_hll_serial,
                          const double flops_parallel, const double flops_parallel_simd, const double flops_parallel_hll, const double flops_parallel_hll_simd, const char *output_file) {
    FILE *fp = fopen(output_file, "a+");  // Append + read

    if (fp == NULL) {
        printf("Errore nell'apertura del file %s\n", output_file);
        return;
    }

    // Verifica se il file è vuoto per decidere se scrivere l'intestazione
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    if (size == 0) {
        // Scrivi l'intestazione se il file è nuovo o vuoto
        fprintf(fp, "matrix_name,rows,cols,nonzeros,num_threads,"
                    "time_serial,time_serial_hll,time_parallel,time_parallel_simd,time_parallel_hll,time_parallel_hll_simd,"
                    "error_csr_relative,error_csr_absolute,error_hll_relative,error_hll_absolute,error_csr_simd_relative,error_csr_simd_absolute,"
                	"error_hll_simd_relative,error_hll_simd_absolute,"
                    "flops_serial,flops_serial_hll,flops_parallel,flops_parallel_simd,flops_parallel_hll,flops_parallel_hll_simd,"
                    "speedup_parallel,speedup_simd,speedup_hll,speedup_hll_simd,"
                    "efficiency_parallel,efficiency_simd,efficiency_hll,efficiency_hll_simd\n");
    }

    // Scrivi una riga di dati
    fprintf(fp,
    "%s,%d,%d,%d,%d,"
    "%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,"
    "%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,"
    "%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
            matrix_name, num_rows, num_cols, nz, num_threads,
            time_serial, time_serial_hll, time_parallel, time_parallel_simd, time_parallel_hll, time_parallel_hll_simd,
            error_csr.mean_rel_err, error_csr.mean_abs_err, error_hll.mean_rel_err, error_hll.mean_abs_err,
            error_csr_simd.mean_rel_err, error_csr_simd.mean_abs_err, error_hll_simd.mean_rel_err, error_hll_simd.mean_abs_err,
            flops_serial, avg_flops_hll_serial, flops_parallel, flops_parallel_simd, flops_parallel_hll, flops_parallel_hll_simd,
            speedup_parallel, speedup_simd, speedup_hll, speedup_hll_simd,
            efficiency_parallel, efficiency_simd, efficiency_hll, efficiency_hll_simd);

    fclose(fp);
}

// Funzione helper per svuotare la cache
void clear_cache(size_t clear_size_mb) {
    size_t bytes_to_clear = clear_size_mb * 1024 * 1024;

    char *dummy_buffer = (char*)malloc(bytes_to_clear);
    if (dummy_buffer == NULL) {
        perror("Failed to allocate cache clearing buffer");
        return; // Continua senza svuotare se fallisce
    }

    printf("Clearing cache with %zu MB buffer...\n", clear_size_mb);

    size_t cache_line_size = 64;
    for (size_t i = 0; i < bytes_to_clear; i += cache_line_size) {
        dummy_buffer[i] = (char)(i % 256);
    }

    FREE_CHECK(dummy_buffer);
    printf("Cache clearing attempt finished.\n");
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

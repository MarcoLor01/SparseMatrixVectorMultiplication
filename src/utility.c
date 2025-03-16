//
// Created by HP on 04/03/2025.
//

#include <stdio.h>
#include "utility.h"
#include <unistd.h>
#include <omp.h>
#include <math.h>
#include <stdbool.h>

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;  // Se OpenMP non è supportato
#endif
}


// Funzioni di utilità per l'ordinamento
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void swap_double(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

// Partizione per quicksort
size_t partition(int *col_idx, double *values, size_t low, size_t high) {
    int pivot = col_idx[high];
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



// Quicksort iterativo per evitare overflow dello stack
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
            size_t pi = partition(col_idx, values, low, high);
            if (pi > 0) sort_row(col_idx, values, low, pi - 1);
            sort_row(col_idx, values, pi + 1, high);
        }
    }
}


double checkDifferences(const double *y_h, const double *y_SerialResult, const int matrix_row, bool full_print) {
    double totalRelativeDiff = 0.0;  // Somma delle differenze relative
    int count = 0;  // Conta gli errori significativi

    // Ciclo su tutte le righe
    for (int i = 0; i < matrix_row; i++) {
        const double toleranceRel = 1e-6;
        const double absTolerance = 1e-7;
        const double currentDiff = fabs(y_SerialResult[i] - y_h[i]);  // Differenza assoluta
        double maxAbs = fmax(fabs(y_SerialResult[i]), fabs(y_h[i]));  // Max tra i valori assoluti

        // Se il massimo valore è piccolo, consideriamo una tolleranza relativa
        if (maxAbs < toleranceRel) {
            maxAbs = toleranceRel;  // Impostiamo un minimo valore per maxAbs
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

void write_results_to_csv(const char *matrix_name, int num_rows, int num_cols, int nz,
                          int num_threads, double time_serial, double time_serial_hll, double time_parallel,
                          double time_parallel_unrolling, double time_parallel_hll, double time_parallel_hll_simd, double speedup_parallel,
                          double speedup_unrolling, double speedup_hll, double speedup_hll_simd, double efficiency_parallel,
                          double efficiency_unrolling, double efficiency_hll, double efficiency_hll_simd, double flops_serial, double flops_hll_serial,
                          double flops_parallel, double flops_parallel_unrolling, double flops_parallel_hll, double flops_parallel_hll_simd, const char *output_file) {
    FILE *fp;
    int file_exists = access(output_file, F_OK) != -1;

    fp = fopen(output_file, "a");
    if (fp == NULL) {
        printf("Errore nell'apertura del file %s\n", output_file);
        return;
    }

    // Scrivi l'intestazione se il file è nuovo
    if (!file_exists) {
        fprintf(fp, "matrix_name,rows,cols,nonzeros,num_threads,"
                    "time_serial,time_serial_hll,time_parallel,time_parallel_unrolling,time_parallel_hll,time_parallel_hll_simd,"
                    "flops_serial,flops_serial_hll,flops_parallel,flops_parallel_unrolling,flops_parallel_hll,flops_parallel_hll_simd,"
                    "speedup_parallel,speedup_unrolling,speedup_hll,speedup_hll_simd,"
                    "efficiency_parallel,efficiency_unrolling,efficiency_hll,efficiency_hll_simd\n");
    }

    // Scrivi una riga di dati
    fprintf(fp, "%s,%d,%d,%d,%d,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
            matrix_name, num_rows, num_cols, nz, num_threads,
            time_serial, time_serial_hll, time_parallel, time_parallel_unrolling, time_parallel_hll, time_parallel_hll_simd,
            flops_serial, flops_hll_serial, flops_parallel, flops_parallel_unrolling, flops_parallel_hll, flops_parallel_hll_simd,
            speedup_parallel, speedup_unrolling, speedup_hll, speedup_hll_simd,
            efficiency_parallel, efficiency_unrolling, efficiency_hll, efficiency_hll_simd);

    fclose(fp);
}




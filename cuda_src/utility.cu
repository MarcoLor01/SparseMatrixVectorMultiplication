//
// Created by HP on 04/03/2025.
//

#include <stdio.h>
#include "utility.h"
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
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


bool checkDifferences(
    const double* ref,
    const double* res,
    int n,
    double abs_tol = 1e-5, // Tolleranza assoluta di default (più stringente)
    double rel_tol = 1e-4, // Tolleranza relativa di default (più stringente)
    bool print_summary = true,
    bool print_first_failure = false)
{
    // Caso base: nessun elemento da confrontare
    if (n <= 0) {
        if (print_summary) {
            printf("--- Comparison Summary ---\n");
            printf("Vector size        : 0\n");
            printf("Result             : PASS (empty vectors)\n");
            printf("------------------------\n");
        }
        return true;
    }

    // Variabili per tracciare gli errori
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    int    diff_count = 0;
    int    first_failure_idx = -1; // Indice del primo elemento che fallisce il test

    // Itera su tutti gli elementi
    for (int i = 0; i < n; ++i) {
        // Calcola la differenza assoluta
        double abs_diff = std::fabs(ref[i] - res[i]);
        double ref_abs = std::fabs(ref[i]);

        // Aggiorna l'errore massimo assoluto
        if (abs_diff > max_abs_err) {
            max_abs_err = abs_diff;
        }

        // Calcola e aggiorna l'errore relativo massimo, solo se il riferimento non è troppo vicino a zero
        // L'errore relativo perde significato o causa divisione per zero se ref[i] è molto piccolo
        if (ref_abs > abs_tol) { // Usiamo abs_tol come soglia per considerare il riferimento non-zero
            double rel_diff = abs_diff / ref_abs;
             if (rel_diff > max_rel_err) {
                max_rel_err = rel_diff;
            }
        }
        // Verifica se l'elemento corrente supera la tolleranza combinata
        // Logica: la differenza è accettabile se è minore della tolleranza assoluta
        // O se è minore della tolleranza relativa *moltiplicata* per la grandezza del riferimento.
        // Formula compatta: absolute(ref - res) <= (abs_tol + rel_tol * absolute(ref))
        if (abs_diff > (abs_tol + rel_tol * ref_abs)) {
            // Errore: l'elemento supera la tolleranza combinata
            diff_count++;
            // Memorizza l'indice del primo errore trovato
            if (first_failure_idx == -1) {
                first_failure_idx = i;
            }
        }
    }

    // Determina l'esito del confronto
    bool passed = (diff_count == 0);

    // Stampa un riepilogo
    if (print_summary) {
        printf("--- Comparison Summary ---\n");
        printf("Vector size        : %d\n", n);
        printf("Absolute Tolerance : %.3e\n", abs_tol);
        printf("Relative Tolerance : %.3e\n", rel_tol);
        printf("Max Absolute Error : %.10e\n", max_abs_err);
        printf("Max Relative Error : %.10e (calculated where |ref| > %.1e)\n", max_rel_err, abs_tol);
        printf("Elements exceeding : %d (%.4f %%)\n", diff_count, n > 0 ? (double)diff_count * 100.0 / n : 0.0);
        printf("Result             : %s\n", passed ? "PASS" : "FAIL");
        // Se fallito e richiesto, stampa i dettagli del primo errore
        if (!passed && print_first_failure && first_failure_idx != -1) {
             printf("------------------------\n");
             printf("First failure at index %d:\n", first_failure_idx);
             printf("  Reference: %.15f\n", ref[first_failure_idx]);
             printf("  Result   : %.15f\n", res[first_failure_idx]);
             double first_fail_abs_diff = std::fabs(ref[first_failure_idx] - res[first_failure_idx]);
             double first_fail_ref_abs = std::fabs(ref[first_failure_idx]);
             printf("  Abs Diff : %.10e\n", first_fail_abs_diff);
             // Stampa errore relativo solo se il riferimento non era troppo piccolo
             if (first_fail_ref_abs > abs_tol) {
                printf("  Rel Diff : %.10e\n", first_fail_abs_diff / first_fail_ref_abs);
             } else {
                printf("  Rel Diff : (not computed, |ref| <= abs_tol)\n");
             }
             printf("  Tolerance: %.10e\n", (abs_tol + rel_tol * first_fail_ref_abs));

        }
        printf("------------------------\n");
    }

    // Restituisce true se nessun elemento ha superato la tolleranza, false altrimenti
    return passed;
}

void write_results_to_csv(const char *matrix_name, const int num_rows, const int num_cols, const int nz,
                          const double time_serial, const double time_serial_hll, const double time_row_csr,
                          const double time_warp_csr, const double time_warp_csr_shared, const double time_warp_shared_hll,
                          const double time_row_hll, const double time_warp_hll,
                          const double flops_serial, const double avg_flops_hll_serial, const double flops_row_csr,
                          const double flops_warp_csr, const double flops_row_hll, const double flops_warp_hll, const double flops_warp_csr_shared,
                          const double flops_warp_shared_hll,
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
                    "time_serial,time_serial_hll,time_row_csr,time_warp_csr,time_warp_shared_csr, time_row_hll,time_warp_hll,time_warp_shared_hll,"
                    "flops_serial,avg_flops_hll_serial,flops_row_csr,flops_warp_csr,flops_warp_csr_shared,flops_row_hll,flops_warp_hll,flops_warp_shared_hll\n");
    }

    // Scrivi una riga di dati
    fprintf(fp, "%s,%d,%d,%d,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
            matrix_name, num_rows, num_cols, nz,
            time_serial, time_serial_hll, time_row_csr, time_warp_csr, time_warp_csr_shared, time_row_hll, time_warp_hll, time_warp_shared_hll,
            flops_serial, avg_flops_hll_serial, flops_row_csr, flops_warp_csr, flops_warp_csr_shared, flops_row_hll, flops_warp_hll, flops_warp_shared_hll);
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




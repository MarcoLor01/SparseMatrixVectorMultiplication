#include <stdio.h>
#include <stdlib.h>
#include "csr_matrix.h"

#include <omp.h>
#include <string.h>
#include <immintrin.h>
#include <utility.h>

void init_csr_matrix(CSRMatrix *mat) {
    mat->M = 0;
    mat->N = 0;
    mat->nz = 0;
    mat->row_ptr = NULL;
    mat->col_idx = NULL;
    mat->values = NULL;
}

void free_csr_matrix(CSRMatrix *mat) {
    free(mat->row_ptr);
    free(mat->col_idx);
    free(mat->values);
    init_csr_matrix(mat);
}

int convert_in_csr(const PreMatrix *pre, CSRMatrix *csr) {
    init_csr_matrix(csr);
    csr->M = pre->M;
    csr->N = pre->N;
    csr->nz = pre->nz;
    memcpy(csr->type, pre->type, sizeof(MM_typecode));

    csr->row_ptr = (int *)calloc(csr->M + 1, sizeof(int)); //Row_ptr contiene gli elementi per riga
    csr->col_idx = (int *)malloc(csr->nz * sizeof(int));
    csr->values = (double *)malloc(csr->nz * sizeof(double));

    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        printf("Errore di allocazione memoria nella conversione CSR\n");
        free(csr->row_ptr);
        free(csr->col_idx);
        free(csr->values);
        return -1;
    }

    // Calcola la memoria occupata in MB
    double memory_size_bytes = 0.0;
    memory_size_bytes += (csr->M + 1) * sizeof(int);  // row_ptr
    memory_size_bytes += csr->nz * sizeof(int);       // col_idx
    memory_size_bytes += csr->nz * sizeof(double);    // values
    memory_size_bytes += sizeof(CSRMatrix);           // struct itself

    double memory_size_mb = memory_size_bytes / (1024.0 * 1024.0);

    printf("Memoria occupata dalla matrice CSR: %.2f MB\n", memory_size_mb);

    // Conta gli elementi per riga
    for (int i = 0; i < pre->nz; i++) {
        csr->row_ptr[pre->I[i] + 1]++;
    }

    // Calcola gli offset cumulativi
    for (int i = 1; i <= csr->M; i++) {
        csr->row_ptr[i] += csr->row_ptr[i - 1];
    }

    // Copia gli elementi nelle posizioni corrette
    int *row_counter = calloc(csr->M, sizeof(int));
    if (!row_counter) {
        printf("Errore di allocazione memoria per row_counter\n");
        free(csr->row_ptr);
        free(csr->col_idx);
        free(csr->values);
        return -1;
    }

    for (int i = 0; i < pre->nz; i++) {
        const int row = pre->I[i];
        const int dest = csr->row_ptr[row] + row_counter[row];
        csr->col_idx[dest] = pre->J[i];
        csr->values[dest] = pre->val[i];
        row_counter[row]++;
    }

    free(row_counter);

    // Ordina gli elementi di ciascuna riga per indice di colonna
    for (int i = 0; i < csr->M; i++) {
        const int start = csr->row_ptr[i];
        const int end = csr->row_ptr[i + 1];

        // Se ci sono elementi da ordinare nella riga
        if (end - start > 1) {
            // Usa la funzione quicksort ottimizzata
            sort_row(csr->col_idx, csr->values, start, end - 1);
        }
    }

    return 0;
}

//Funzione sequenziale per operazione spVM

void csr_matrix_vector_mult(const int num_row,
    const int *row_ptr, const int *col_idx, const double *values,
    const double *x, double *y) {

    for (int i = 0; i < num_row; i++) {

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y[i] += values[j] * x[col_idx[j]]; // Accumulo in registro
        }
    }
}

void print_csr_matrix(const CSRMatrix *mat) {
    printf("Dimensioni matrice: %d x %d\n", mat->M, mat->N);
    printf("Numero di elementi non-zero: %d\n", mat->nz);
    printf("Matrix type: %s\n", mm_typecode_to_str(mat->type));
    if(mat->M <= 30 && mat->N <= 30) {
        printf("row_ptr: ");
        for (int i = 0; i <= mat->M; i++) {
            printf("%d ", mat->row_ptr[i]);
        }
        printf("\n");

        printf("col_idx: ");
        for (int i = 0; i < mat->nz; i++) {
            printf("%d ", mat->col_idx[i]);
        }
        printf("\n");

        printf("values: ");
        for (int i = 0; i < mat->nz; i++) {
            printf("%f ", mat->values[i]);
        }
        printf("\n");
    }
}

void spvm_csr_parallel(const int *row_ptr, const int *col_idx, const double *values, const double *x,
    double *y, int num_threads, const int *thread_row_start, const int *thread_row_end) {
    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();

        // Compute phase
        for (int i = thread_row_start[thread_id]; i < thread_row_end[thread_id]; i++) {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];

            // Uso variabile temporanea per migliorare la località di cache
            double temp_sum = 0.0;
            for (int j = row_start; j < row_end; j++) {
                temp_sum += values[j] * x[col_idx[j]];
            }
            y[i] = temp_sum; // Assegnazione single-write
        }
    }
}


/* Ogni thread dovrebbe avere un numero simile di elementi da elaborare.
     * Questa funzione calcola la distribuzione delle righe tra i thread, basandosi sul numero di nz,
     * in modo che il numero di elementi non sia troppo sbilanciato.
     * Restituisce il numero effettivo di thread utilizzati.
*/

int prepare_thread_distribution(const int num_row, const int *row_ptr, int num_threads,
                               const long long total_nnz, int **thread_row_start, int **thread_row_end) {
    // Check special cases and allocate memory
    if (num_row <= 0 || num_threads <= 0) {
        return 0;
    }

    // Use fewer threads if there are fewer rows than threads
    if (num_row < num_threads) {
        num_threads = num_row;
    }

    // Allocate memory
    *thread_row_start = malloc((size_t)num_threads * sizeof(int));
    *thread_row_end = malloc((size_t)num_threads * sizeof(int));
    int *thread_nnz = malloc((size_t)num_threads * sizeof(int));

    if (*thread_row_start == NULL || *thread_row_end == NULL || thread_nnz == NULL) {
        // Handle memory allocation failure
        if (*thread_row_start) free(*thread_row_start);
        if (*thread_row_end) free(*thread_row_end);
        if (thread_nnz) free(thread_nnz);
        return 0;
    }

    // Initialize arrays
    for (int t = 0; t < num_threads; t++) {
        (*thread_row_start)[t] = -1;
        (*thread_row_end)[t] = -1;
        thread_nnz[t] = 0;
    }

    // Calculate target non-zeros per thread (ceiling division)
    const long long nnz_per_thread = (total_nnz + num_threads - 1) / num_threads;

    // Single-pass distribution of rows based on non-zero elements
    int current_thread = 0;
    int current_nnz = 0;

    for (int i = 0; i < num_row; i++) {
        int row_nnz = row_ptr[i+1] - row_ptr[i];

        // Set start row for current thread if not set yet
        if ((*thread_row_start)[current_thread] == -1) {
            (*thread_row_start)[current_thread] = i;
        }

        current_nnz += row_nnz;
        thread_nnz[current_thread] += row_nnz;

        // If we've reached the target for this thread and it's not the last thread
        if (current_nnz >= nnz_per_thread && current_thread < num_threads - 1) {
            (*thread_row_end)[current_thread] = i + 1; // End is exclusive
            current_thread++;
            current_nnz = 0;
        }
    }

    // Ensure the last active thread covers all remaining rows
    if (current_thread < num_threads) {
        (*thread_row_end)[current_thread] = num_row;
        current_thread++;
    }

    // Determine how many threads are actually used
    int valid_threads = 0;
    for (int t = 0; t < num_threads; t++) {
        if ((*thread_row_start)[t] != -1 && (*thread_row_end)[t] != -1 && thread_nnz[t] > 0) {
            // Keep valid thread data by compacting the arrays
            if (valid_threads != t) {
                (*thread_row_start)[valid_threads] = (*thread_row_start)[t];
                (*thread_row_end)[valid_threads] = (*thread_row_end)[t];
                thread_nnz[valid_threads] = thread_nnz[t];
            }
            valid_threads++;
        }
    }

    // Update the actual number of threads used
    num_threads = valid_threads;

    // Only do one pass to avoid excessive overhead
    for (int t = 0; t < num_threads - 1; t++) {
        // Check if next thread has significantly more work (more than 1.5x)
        if (thread_nnz[t+1] > thread_nnz[t] * 1.5) {
            int start_row = (*thread_row_start)[t+1];
            int row_nnz = row_ptr[start_row+1] - row_ptr[start_row];

            // Only move the row if it doesn't make the current thread overloaded
            if (thread_nnz[t] + row_nnz <= nnz_per_thread * 1.2) {
                // Move one row from next thread to current thread
                (*thread_row_end)[t]++;
                (*thread_row_start)[t+1]++;

                // Update non-zero counts
                thread_nnz[t] += row_nnz;
                thread_nnz[t+1] -= row_nnz;
            }
        }
    }

    // Add detailed logging for each thread
    printf("\n--- Dettagli distribuzione thread ---\n");
    printf("Thread attivi: %d (su %d richiesti inizialmente)\n", valid_threads, num_threads);
    printf("Numero totale di nnz: %lld\n", total_nnz);
    printf("Numero totale di righe: %d\n\n", num_row);

    for (int t = 0; t < num_threads; t++) {
        int num_rows = (*thread_row_end)[t] - (*thread_row_start)[t];
        int actual_nnz = 0;

        // Calcola il numero effettivo di nnz per questo thread
        for (int i = (*thread_row_start)[t]; i < (*thread_row_end)[t]; i++) {
            actual_nnz += row_ptr[i+1] - row_ptr[i];

        }

        printf("Thread %d: %d righe (da %d a %d), %d nnz (%.2f%% del totale)\n",
                t,
                num_rows,
                (*thread_row_start)[t],
                (*thread_row_end)[t] - 1,
                actual_nnz,
                (actual_nnz * 100.0) / total_nnz);
    }
    printf("--- Fine dettagli distribuzione ---\n\n");
    free(thread_nnz);
    return valid_threads;
}


void spvm_csr_parallel_simd(const int *row_ptr, const int *col_idx, const double *values,
                                     const double *x, double *y, int num_threads,
                                     const int *thread_row_start, const int *thread_row_end) {
    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int start_row = thread_row_start[thread_id];
        const int end_row = thread_row_end[thread_id];

        for (int i = start_row; i < end_row; i++) {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];

            double temp_sum = 0.0;

            #pragma omp simd reduction(+:temp_sum)
            for (int j = row_start; j < row_end; j++) {
                temp_sum += values[j] * x[col_idx[j]];
            }

            y[i] = temp_sum;
        }
    }
}


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

    // Conta gli elementi per riga
    for (int i = 0; i < pre->nz; i++) {
        csr->row_ptr[pre->I[i] + 1]++;
    }

    // Calcola gli offset cumulativi, prima ogni elemento conteneva
    // il numero di elementi della riga, ora contiene l'indice del primo elemento
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
            // Usa la tua funzione quicksort ottimizzata
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
                               const long long total_nnz, int *thread_row_start, int *thread_row_end) {
    // Check special cases
    if (num_row <= 0 || num_threads <= 0) {
        return 0;
    }

    // Use fewer threads if there are fewer rows than threads
    if (num_row < num_threads) {
        num_threads = num_row;
    }

    // Calculate target non-zeros per thread
    const long long nnz_per_thread = (total_nnz + num_threads - 1) / num_threads; // Ceiling division

    // Initialize
    long long current_thread_nnz = 0;
    int current_thread = 0;
    thread_row_start[0] = 0;

    // First pass: distribute rows based on non-zero elements
    for (int i = 0; i < num_row; i++) {
        const int row_nnz = row_ptr[i+1] - row_ptr[i];

        // If adding this row would exceed the target and we haven't assigned the last thread
        if (current_thread_nnz + row_nnz > nnz_per_thread &&
            current_thread < num_threads - 1 &&
            i > thread_row_start[current_thread]) { // Ensure at least one row per thread

            thread_row_end[current_thread] = i;
            current_thread++;
            thread_row_start[current_thread] = i;
            current_thread_nnz = 0;
        }

        current_thread_nnz += row_nnz;
    }

    // Make sure all threads have been assigned work
    while (current_thread < num_threads - 1) {
        // If we have fewer distinct workloads than threads, some threads might not get work
        // In this case, just divide the remaining rows evenly
        int remaining_rows = num_row - thread_row_start[current_thread];
        int rows_per_remaining_thread = remaining_rows / (num_threads - current_thread);

        if (rows_per_remaining_thread > 0) {
            thread_row_end[current_thread] = thread_row_start[current_thread] + rows_per_remaining_thread;
            current_thread++;
            thread_row_start[current_thread] = thread_row_end[current_thread-1];
        } else {
            // Can't subdivide further
            break;
        }
    }

    // Last thread gets all remaining rows
    thread_row_end[current_thread] = num_row;

    // Calculate actual thread count used
    int actual_threads_used = current_thread + 1;

    // Second pass: balance the load by checking if any thread has significantly more work
    long long *thread_nnz = (long long *)malloc(actual_threads_used * sizeof(long long));
    if (!thread_nnz) {
        return actual_threads_used; // Can't balance, but return how many threads we're using
    }

    // Calculate non-zeros per thread
    for (int t = 0; t < actual_threads_used; t++) {
        thread_nnz[t] = 0;
        for (int i = thread_row_start[t]; i < thread_row_end[t]; i++) {
            thread_nnz[t] += (row_ptr[i+1] - row_ptr[i]);
        }
    }

    // Find imbalances and correct them
    for (int t = 0; t < actual_threads_used - 1; t++) {
        // If next thread has much more work
        if (thread_nnz[t+1] > thread_nnz[t] * 1.5) {
            // Move rows from t+1 to t until balanced or no more rows can be moved
            while (thread_row_end[t] < thread_row_start[t+1] + 1 &&
                   thread_nnz[t+1] > thread_nnz[t] * 1.2) {

                int row_to_move = thread_row_start[t+1];
                int row_nnz = row_ptr[row_to_move+1] - row_ptr[row_to_move];

                // Update boundaries
                thread_row_end[t]++;
                thread_row_start[t+1]++;

                // Update non-zero counts
                thread_nnz[t] += row_nnz;
                thread_nnz[t+1] -= row_nnz;
            }
        }
    }

    free(thread_nnz);
    return actual_threads_used;
}


void spvm_csr_parallel_unrolling(const int *row_ptr, const int *col_idx, const double *values, const double *x,
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

            #pragma omp simd reduction(+:temp_sum)
            for (int j = row_start; j < row_end; j++) {
                temp_sum += values[j] * x[col_idx[j]];
            }
            y[i] = temp_sum; // Assegnazione single-write
        }
    }
}



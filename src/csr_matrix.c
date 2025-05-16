#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include <string.h>
#include <immintrin.h>
#include <utility.h>
#include <csr_matrix.h>
#include <matrix_parser.h>

void init_csr_matrix(CSRMatrix *mat) {
    mat->M = 0;
    mat->N = 0;
    mat->nz = 0;
    mat->row_ptr = NULL;
    mat->col_idx = NULL;
    mat->values = NULL;
}

void free_csr_matrix(CSRMatrix *mat) {
    FREE_CHECK(mat->row_ptr);
    FREE_CHECK(mat->col_idx);
    FREE_CHECK(mat->values);
    init_csr_matrix(mat);
}

// Funzione per scrivere i dati della memoria in un file CSV
void write_memory_stats_to_csv(const char* matrix_name, int nz, size_t total_memory_bytes) {
    const char* directory = "../result";
    const char* filename = "matrix_memory_stats_csr.csv";
    char filepath[256];
    int file_exists = 0;

    // Crea il percorso completo del file
    snprintf(filepath, sizeof(filepath), "%s/%s", directory, filename);

    // Verifica se il file esiste già
    FILE* check_file = fopen(filepath, "r");
    if (check_file != NULL) {
        file_exists = 1;
        fclose(check_file);
    }

    FILE* fp = fopen(filepath, "a");
    if (fp == NULL) {
        printf("Errore nell'apertura del file CSV per le statistiche di memoria\n");
        return;
    }

    // Se il file è nuovo, scrivi l'intestazione
    if (!file_exists) {
        fprintf(fp, "Matrix Name,Non-Zero Elements,Memory Size (MB)\n");
    }

    // Scrivi i dati
    double memory_mb = total_memory_bytes / (1024.0 * 1024.0);
    fprintf(fp, "%s,%d,%.4f\n", matrix_name, nz, memory_mb);

    fclose(fp);
    printf("Statistiche di memoria salvate in %s\n", filepath);
}

int convert_in_csr(const PreMatrix *pre, CSRMatrix *csr, const char* matrix_name) {
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
        free_csr_matrix(csr);
        return -1;
    }

    // Calcolo della memoria utilizzata dalla matrice CSR
    size_t row_ptr_size = (csr->M + 1) * sizeof(int);
    size_t col_idx_size = csr->nz * sizeof(int);
    size_t values_size = csr->nz * sizeof(double);
    size_t total_memory = row_ptr_size + col_idx_size + values_size;

    // Conta gli elementi per riga
    for (int i = 0; i < pre->nz; i++) {
        csr->row_ptr[pre->I[i] + 1]++;
    }

    // Calcola gli offset cumulativi
    for (int i = 1; i <= csr->M; i++) {
        csr->row_ptr[i] += csr->row_ptr[i - 1];
    }

    // Copia gli elementi nelle posizioni corrette
    int *row_counter = (int *)calloc(csr->M, sizeof(int));
    if (!row_counter) {
        printf("Errore di allocazione memoria per row_counter\n");
        free_csr_matrix(csr);
        return -1;
    }

    for (int i = 0; i < pre->nz; i++) {
        const int row = pre->I[i];
        const int dest = csr->row_ptr[row] + row_counter[row];
        csr->col_idx[dest] = pre->J[i];
        csr->values[dest] = pre->val[i];
        row_counter[row]++;
    }

    FREE_CHECK(row_counter);

    // Ordina gli elementi di ciascuna riga per indice di colonna
    for (int i = 0; i < csr->M; i++) {
        const int start = csr->row_ptr[i];
        const int end = csr->row_ptr[i + 1];

        // Se ci sono elementi da ordinare nella riga
        if (end - start > 1) {
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
            y[i] += values[j] * x[col_idx[j]]; // Accumulo
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


int prepare_thread_distribution(const int num_row, const int *row_ptr, int num_threads,
                               const long long total_nnz, int **thread_row_start, int **thread_row_end) {
    // Verifiche iniziali e casi speciali
    if (num_row <= 0 || num_threads <= 0) return 0;

    // Limita il numero di thread se ci sono meno righe
    num_threads = (num_row < num_threads) ? num_row : num_threads;

    // Allocazione memoria
    *thread_row_start = malloc((size_t)num_threads * sizeof(int));
    *thread_row_end = malloc((size_t)num_threads * sizeof(int));
    int *thread_nnz = malloc((size_t)num_threads * sizeof(int));

    if (!*thread_row_start || !*thread_row_end || !thread_nnz) {
        // Pulizia in caso di fallimento
        FREE_CHECK(*thread_row_start);
        FREE_CHECK(*thread_row_end);
        FREE_CHECK(thread_nnz);
        return 0;
    }

    // Inizializzazione
    for (int t = 0; t < num_threads; t++) {
        (*thread_row_start)[t] = -1;
        (*thread_row_end)[t] = -1;
        thread_nnz[t] = 0;
    }

    // Calcolo obiettivo nnz per thread
    const long long nnz_per_thread = (total_nnz + num_threads - 1) / num_threads;

    // Distribuzione delle righe in base al carico di lavoro
    int current_thread = 0;
    int current_nnz = 0;

    for (int i = 0; i < num_row; i++) {
        int row_nnz = row_ptr[i+1] - row_ptr[i];

        // Imposta la riga di inizio per questo thread se non è ancora impostata
        if ((*thread_row_start)[current_thread] == -1) {
            (*thread_row_start)[current_thread] = i;
        }

        current_nnz += row_nnz;
        thread_nnz[current_thread] += row_nnz;

        // Passiamo al thread successivo se questo ha raggiunto il target (e non è l'ultimo)
        if (current_nnz >= nnz_per_thread && current_thread < num_threads - 1) {
            (*thread_row_end)[current_thread] = i + 1;  // End è esclusivo
            current_thread++;
            current_nnz = 0;
        }
    }

    // Assicuriamoci che l'ultimo thread attivo copra tutte le righe rimanenti
    if (current_thread < num_threads) {
        (*thread_row_end)[current_thread] = num_row;
        current_thread++;
    }

    // Compattazione per rimuovere i thread non utilizzati
    int valid_threads = 0;
    for (int t = 0; t < num_threads; t++) {
        if ((*thread_row_start)[t] != -1 && (*thread_row_end)[t] != -1 && thread_nnz[t] > 0) {
            if (valid_threads != t) {
                (*thread_row_start)[valid_threads] = (*thread_row_start)[t];
                (*thread_row_end)[valid_threads] = (*thread_row_end)[t];
                thread_nnz[valid_threads] = thread_nnz[t];
            }
            valid_threads++;
        }
    }

    printf("\n--- Dettagli distribuzione thread ---\n");
    printf("Thread attivi: %d (su %d richiesti inizialmente)\n", valid_threads, num_threads);
    printf("Numero totale di nnz: %lld\n", total_nnz);
    printf("Numero totale di righe: %d\n\n", num_row);

    for (int t = 0; t < valid_threads; t++) {
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

    FREE_CHECK(thread_nnz);
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

void spvm_csr_parallel(const int *row_ptr, const int *col_idx, const double *values, const double *x,
    double *y, int num_threads, const int *thread_row_start, const int *thread_row_end) {
    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();

        // Compute phase
        for (int i = thread_row_start[thread_id]; i < thread_row_end[thread_id]; i++) {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];

            // Uso variabile temporanea per avere meno accessi in memoria
            double temp_sum = 0.0;
            for (int j = row_start; j < row_end; j++) {
                temp_sum += values[j] * x[col_idx[j]];
            }
            y[i] = temp_sum; // Assegnazione single-write
        }
    }
}




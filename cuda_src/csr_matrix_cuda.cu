#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>
#include "csr_matrix_cuda.cuh"

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
        free_csr_matrix(csr);
        return -1;
    }

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
            y[i] += values[j] * x[col_idx[j]];
        }
    }
}

void print_csr_matrix(const CSRMatrix *mat) {
    printf("Dimensioni matrice: %d x %d\n", mat->M, mat->N);
    printf("Numero di elementi non-zero: %d\n", mat->nz);

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

// Kernel CUDA Naive per SpMV - un thread per ogni riga
__global__ void spmv_csr_naive_kernel(
    int M,              // Numero di righe
    const int *row_ptr, // Array dei puntatori alle righe
    const int *col_idx, // Array degli indici delle colonne
    const double *values, // Array dei valori
    const double *x,    // Vettore di input
    double *y           // Vettore di output
) {
    // Calcola l'indice della riga da elaborare
    int row = blockIdx.x * blockDim.x + threadIdx.x; //Blocco in cui mi trovo * Thread nel blocco * Thread ID

    // Verifica che l'indice sia valido
    if (row < M) {
        double dot = 0.0;
        // Indice iniziale e finale nella struttura CSR per questa riga
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        // Calcola il prodotto scalare per questa riga
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_idx[i]];
        }

        // Memorizza il risultato
        y[row] = dot;
    }
}

// Kernel CUDA con approccio "un warp per riga" per SpMV
__global__ void spmv_csr_warp_kernel(
    int M,              // Numero di righe
    const int *row_ptr, // Array dei puntatori alle righe
    const int *col_idx, // Array degli indici delle colonne
    const double *values, // Array dei valori
    const double *x,    // Vettore di input
    double *y           // Vettore di output
) {
    // Identificativo del warp all'interno della griglia
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32; //Come precedentemente, ma divido per 32 essendo la dimensione del warp

    // Identificativo del thread all'interno del warp (0-31)
    int lane_id = threadIdx.x % 32;

    // Calcola la riga da elaborare in base all'ID del warp
    int row = warp_id;

    // Verifica che la riga sia valida
    if (row < M) {
        // Indice iniziale e finale nella struttura CSR per questa riga
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        // Risultato parziale per questo thread
        double dot = 0.0;

        // Ogni thread del warp elabora una porzione degli elementi della riga
        for (int i = row_start + lane_id; i < row_end; i += 32) {
            dot += values[i] * x[col_idx[i]];
        }

        // Riduzione parallela all'interno del warp
        #pragma unroll //Consigliamo al compilatore di sviluppare come sequenza di istruzioni lineari
                       // invece di un ciclo con controllo di flusso
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset); //Ogni thread ha una parte del risultato finale, sommo
            //Ora con offset di 16 andando a ridurre ogni volta, avrò quindi ad es. nel primo passo thr. 0 + thr. 16,
            //nel secondo thread.0 + thread.8, ecc. fino ad avere nel thread 0 tutto il risultato accumulato.
        }

        // Solo il primo thread del warp (lane_id == 0) scrive il risultato, perchè ora possiede tutto
        if (lane_id == 0) {
            y[row] = dot;
        }
    }
}

__global__ void spmv_csr_warp_shared_memory_kernel(
    int M, const int *row_ptr, const int *col_idx, const double *values,
    const double *x, double *y, int cacheSize
) {
    extern __shared__ double s_x[];
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = threadIdx.x % 32;

    if(warp_id >= M) return;

    // Carica collaborativamente i valori di x nella memoria condivisa

    for(int i = threadIdx.x; i < cacheSize; i += blockDim.x) {
        s_x[i] = x[i]; //Inserisco il valore di x nelle posizione riservate al thread, che saranno quindi nel punto iniziale
        //l'indice del thread nel blocco, e poi andremo avanti per il numero di thread nel blocco.
    }

    // Sincronizza tutti i thread nel blocco
    __syncthreads();

    int row_start = row_ptr[warp_id];
    int row_end = row_ptr[warp_id + 1];
    double sum = 0.0;

    // Calcolo SpMV
    for(int i = row_start + lane_id; i < row_end; i += 32) {
        int col = col_idx[i];
        double val = values[i];

        if(col < MAX_CACHE) {
            sum += val * s_x[col];
        } else {
            sum += val * x[col];
        }
    }

    // Riduzione e scrittura
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if(lane_id == 0) y[warp_id] = sum;
}
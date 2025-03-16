#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "hll_matrix.h"

#include <immintrin.h>

#include "utility.h"

void init_hll_matrix(HLLMatrix *hll) {
    hll->num_blocks = 0;
    hll->blocks = NULL;
}

int convert_in_hll(const PreMatrix *pre, HLLMatrix *hll) {
    // Inizializzazione della matrice HLL
    init_hll_matrix(hll);

    // Controllo input valido
    if (!pre || !hll) {
        printf("Errore: Parametri non validi\n");
        return -1;
    }

    // Calcolo del numero di blocchi
    const int M = pre->M;
    const int num_blocks = (M + HACK_SIZE - 1) / HACK_SIZE; // Arrotondamento per eccesso
    hll->num_blocks = num_blocks;

    // Allocazione dei blocchi
    hll->blocks = (ELLPACKBlock *)calloc(num_blocks, sizeof(ELLPACKBlock));
    if (!hll->blocks) {
        printf("Errore di allocazione memoria per i blocchi HLL\n");
        free_hll_matrix(hll);
        return -1;
    }

    // Calcolo del numero di elementi non zero per riga
    int *row_counts = calloc(M, sizeof(int));
    if (!row_counts) {
        printf("Errore di allocazione memoria per row_counts\n");
        free_hll_matrix(hll);
        return -1;
    }

    // Conteggio elementi per riga
    for (int i = 0; i < pre->nz; i++) {
        if (pre->I[i] >= 0 && pre->I[i] < M) {
            row_counts[pre->I[i]]++; //Quanti elementi NZ ci sono per riga
        } else {
            printf("ERRORE: Indice di riga non valido: %d (max: %d)\n", pre->I[i], M-1);
            free(row_counts);
            free_hll_matrix(hll);
            return -1;
        }
    }

    // Inizializzazione dei blocchi
    for (int b = 0; b < num_blocks; b++) {
        const int start_row = b * HACK_SIZE;
        const int end_row = b == num_blocks - 1 ? M : start_row + HACK_SIZE;
        const int num_rows = end_row - start_row;

        // Calcola MAXNZ per questo blocco
        int max_nnz = 0;
        for (int i = start_row; i < end_row && i < M; i++) {
            if (row_counts[i] > max_nnz) {
                max_nnz = row_counts[i];
            }
        }

        // Inizializza il blocco
        hll->blocks[b].M = num_rows;
        hll->blocks[b].N = pre->N;
        hll->blocks[b].MAXNZ = max_nnz;

        // Allocazione degli array JA e AS in formato trasposto
        if (max_nnz > 0) {
            // Allocazione linearizzata di JA (MAXNZ x num_rows)
            hll->blocks[b].JA = (int *)malloc(max_nnz * num_rows * sizeof(int));
            if (!hll->blocks[b].JA) {
                printf("ERRORE: Allocazione fallita per JA del blocco %d\n", b);
                free_hll_matrix(hll);
                free(row_counts);
                return -1;
            }

            // Allocazione linearizzata di AS (MAXNZ x num_rows)
            hll->blocks[b].AS = (double *)malloc(max_nnz * num_rows * sizeof(double));
            if (!hll->blocks[b].AS) {
                printf("ERRORE: Allocazione fallita per AS del blocco %d\n", b);
                free_hll_matrix(hll);
                free(row_counts);
                return -1;
            }

            // Inizializzazione degli array
            for (int i = 0; i < max_nnz * num_rows; i++) {
                hll->blocks[b].JA[i] = -1;
                hll->blocks[b].AS[i] = 0.0;
            }
        } else {
            // Blocco vuoto
            hll->blocks[b].JA = NULL;
            hll->blocks[b].AS = NULL;
        }
    }

    // Reset contatori per riempimento
    int *current_pos = (int *)calloc(M, sizeof(int));
    if (!current_pos) {
        printf("Errore di allocazione memoria per current_pos\n");
        free_hll_matrix(hll);
        free(row_counts);
        return -1;
    }

    // Riempimento degli array HLL
    for (int i = 0; i < pre->nz; i++) {
        int row = pre->I[i];
        int col = pre->J[i];
        double val = pre->val[i];

        // Verifica validità indici
        if (row < 0 || row >= M || col < 0 || col >= pre->N) {
            printf("AVVISO: Indici non validi: riga=%d, colonna=%d\n", row, col);
            continue;
        }

        // Calcolo indici e posizioni
        int block_idx = row / HACK_SIZE;
        int local_row = row % HACK_SIZE;
        int pos = current_pos[row];

        // Verifica validità blocco
        if (block_idx >= num_blocks) {
            printf("ERRORE: Blocco non valido: %d (max: %d)\n", block_idx, num_blocks-1);
            continue;
        }

        // Verifica validità riga locale
        if (local_row >= hll->blocks[block_idx].M) {
            printf("ERRORE: Riga locale non valida: %d (max: %d)\n", local_row, hll->blocks[block_idx].M-1);
            continue;
        }

        // Verifica che ci sia spazio per l'elemento
        if (pos >= hll->blocks[block_idx].MAXNZ) {
            printf("ERRORE: Troppe entrate per la riga %d (max: %d)\n", row, hll->blocks[block_idx].MAXNZ);
            continue;
        }

        // Inserimento valore con layout trasposto: pos * num_rows + local_row
        int idx = pos * hll->blocks[block_idx].M + local_row;

        hll->blocks[block_idx].JA[idx] = col;
        hll->blocks[block_idx].AS[idx] = val;
        current_pos[row]++;
    }

    // Ordinamento degli elementi per indice di colonna per ogni riga
    // Questo è più complesso con layout trasposto
    for (int b = 0; b < num_blocks; b++) {
        if (!hll->blocks[b].JA) continue; // Salta blocchi vuoti

        const int num_rows = hll->blocks[b].M;
        const int max_nnz = hll->blocks[b].MAXNZ;

        for (int i = 0; i < num_rows; i++) {
            int global_row = b * HACK_SIZE + i;

            // Salta righe fuori matrice
            if (global_row >= M) continue;

            // Estrai gli elementi di questa riga
            int *row_JA = (int *)malloc(max_nnz * sizeof(int));
            double *row_AS = (double *)malloc(max_nnz * sizeof(double));

            if (!row_JA || !row_AS) {
                printf("ERRORE: Allocazione fallita per gli array temporanei\n");
                if (row_JA) free(row_JA);
                if (row_AS) free(row_AS);
                free_hll_matrix(hll);
                free(row_counts);
                free(current_pos);
                return -1;
            }

            // Copia gli elementi in array temporanei
            int count = 0;
            for (int j = 0; j < max_nnz; j++) {
                int idx = j * num_rows + i;
                if (hll->blocks[b].JA[idx] != -1) {
                    row_JA[count] = hll->blocks[b].JA[idx];
                    row_AS[count] = hll->blocks[b].AS[idx];
                    count++;
                }
            }

            // Insertion sort
            for (int j = 1; j < count; j++) {
                int temp_col = row_JA[j];
                double temp_val = row_AS[j];
                int k = j;

                while (k > 0 && row_JA[k-1] > temp_col) {
                    row_JA[k] = row_JA[k-1];
                    row_AS[k] = row_AS[k-1];
                    k--;
                }

                row_JA[k] = temp_col;
                row_AS[k] = temp_val;
            }

            // Ripopolamento della riga ordinata
            for (int j = 0; j < max_nnz; j++) {
                int idx = j * num_rows + i;
                if (j < count) {
                    hll->blocks[b].JA[idx] = row_JA[j];
                    hll->blocks[b].AS[idx] = row_AS[j];
                } else {
                    hll->blocks[b].JA[idx] = -1;
                    hll->blocks[b].AS[idx] = 0.0;
                }
            }

            free(row_JA);
            free(row_AS);
        }
    }

    // Calcolo memoria utilizzata
    size_t total_memory = 0;
    double max_block_mem_gb = 0.0;

    for (int b = 0; b < num_blocks; b++) {
        if (hll->blocks[b].MAXNZ > 0) {
            size_t block_mem = (size_t)hll->blocks[b].M * hll->blocks[b].MAXNZ * (sizeof(int) + sizeof(double));
            total_memory += block_mem;

            double block_mem_gb = block_mem / (1024.0 * 1024.0 * 1024.0);
            if (block_mem_gb > max_block_mem_gb) {
                max_block_mem_gb = block_mem_gb;
            }
        }
    }

    double total_mem_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
    printf("Memoria HLL totale: %.5f GB, Memoria massima per blocco: %.5f GB\n",
           total_mem_gb, max_block_mem_gb);

    // Pulizia e ritorno
    free(row_counts);
    free(current_pos);
    return 0;
}

// Funzione per liberare la memoria di una matrice HLL con layout trasposto
void free_hll_matrix(HLLMatrix *hll) {
    if (!hll) return;

    // Liberazione della memoria per ciascun blocco
    if (hll->blocks) {
        for (int b = 0; b < hll->num_blocks; b++) {
            // Con il layout trasposto, JA e AS sono array monodimensionali
            if (hll->blocks[b].JA) {
                free(hll->blocks[b].JA);
                hll->blocks[b].JA = NULL;
            }

            if (hll->blocks[b].AS) {
                free(hll->blocks[b].AS);
                hll->blocks[b].AS = NULL;
            }
        }
        free(hll->blocks);
        hll->blocks = NULL;
    }
    // Resetta i valori
    hll->num_blocks = 0;
}

void spmv_hll_simd(HLLMatrix *A, const double *x, double *y, int num_threads,
                        int const *thread_block_start, int const *thread_block_end) {

    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int start_block = thread_block_start[thread_id];
        const int end_block = thread_block_end[thread_id];

        // Calcola la moltiplicazione SpMV per i blocchi assegnati a questo thread
        for (int b = start_block; b < end_block; b++) {
            const ELLPACKBlock *block = &A->blocks[b];
            const int row_offset = b * HACK_SIZE;  // Calcolo dell'offset delle righe globali
            const int num_rows = block->M;

            for (int i = 0; i < num_rows; i++) {
                double sum = 0.0;

                // Utilizziamo la direttiva SIMD per consigliare al compilatore di vettorizzare
                #pragma omp simd reduction(+:sum)
                for (int j = 0; j < block->MAXNZ; j++) {
                    const int idx = j * num_rows + i;
                    const int col = block->JA[idx];
                    if (col >= 0 && col < block->N) {
                        sum += block->AS[idx] * x[col];
                    }
                }

                y[row_offset + i] = sum;
            }
        }
    }
}

// Funzione serializzata per la moltiplicazione SpMV
void spmv_hll_serial(const HLLMatrix *A, const double *x, double *y) {
    // Elabora ogni blocco della matrice HLL
    for (int b = 0; b < A->num_blocks; b++) {
        const ELLPACKBlock *block = &A->blocks[b];
        const int row_offset = b * HACK_SIZE;  // Calcolo dell'offset delle righe globali
        const int num_rows = block->M;

        // Per ogni riga nel blocco corrente
        for (int i = 0; i < num_rows; i++) {
            double sum = 0.0;

            // Per ogni possibile elemento non-zero nella riga
            for (int j = 0; j < block->MAXNZ; j++) {
                // Accesso all'elemento (i,j) nel layout trasposto
                const int idx = j * num_rows + i;
                const int col = block->JA[idx];

                // Verifica che l'indice di colonna sia valido
                if (col >= 0 && col < block->N) {
                    sum += block->AS[idx] * x[col];
                }
            }

            // Inserisce il risultato nella posizione corretta del vettore y
            y[row_offset + i] = sum;
        }
    }
}

void spmv_hll(const HLLMatrix *A, const double *x, double *y, int num_threads,
              int const *thread_block_start, int const *thread_block_end) {

    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int start_block = thread_block_start[thread_id];
        const int end_block = thread_block_end[thread_id];

        // Calcola la moltiplicazione SpMV per i blocchi assegnati a questo thread
        for (int b = start_block; b < end_block; b++) {
            const ELLPACKBlock *block = &A->blocks[b];
            const int row_offset = b * HACK_SIZE;  // Calcolo dell'offset delle righe globali
            const int num_rows = block->M;

            for (int i = 0; i < num_rows; i++) {
                double sum = 0.0;

                for (int j = 0; j < block->MAXNZ; j++) {
                    // Accesso all'elemento (i,j) nel layout trasposto
                    const int idx = j * num_rows + i;
                    const int col = block->JA[idx];
                    if (col >= 0 && col < block->N) {
                        sum += block->AS[idx] * x[col];
                    }
                }

                y[row_offset + i] = sum;  // Inserisce il risultato nella posizione corretta
            }
        }
    }
}

int prepare_thread_distribution_hll(const HLLMatrix *matrix, int num_threads,
                                    int *thread_block_start, int *thread_block_end) {
    // Calcolo del numero totale di elementi non zero effettivi
    long long total_nnz = 0;
    int *nnz_per_block = (int*)malloc(matrix->num_blocks * sizeof(int));

    // Contiamo gli elementi non zero effettivi per ogni blocco
    for (int b = 0; b < matrix->num_blocks; b++) {
        ELLPACKBlock *block = &matrix->blocks[b];
        int block_nnz = 0;
        int num_rows = block->M;

        for (int i = 0; i < num_rows; i++) {
            int count = 0;
            for (int j = 0; j < block->MAXNZ; j++) {
                // Accesso all'elemento (i,j) nel layout trasposto
                int idx = j * num_rows + i;
                if (block->JA[idx] >= 0 && block->JA[idx] < block->N) {
                    count++;
                }
            }

            block_nnz += count;
        }

        nnz_per_block[b] = block_nnz;
        total_nnz += block_nnz;
    }

    // Numero target di elementi non zero per thread
    const long long nnz_per_thread = total_nnz / num_threads;
    long long accumulated_nnz = 0;
    int current_thread = 0;

    thread_block_start[0] = 0;

    for (int b = 0; b < matrix->num_blocks; b++) {
        accumulated_nnz += nnz_per_block[b];

        // Se questo thread ha raggiunto il target e non è l'ultimo thread
        if (accumulated_nnz >= (current_thread + 1) * nnz_per_thread && current_thread < num_threads - 1) {
            thread_block_end[current_thread] = b + 1;
            current_thread++;
            thread_block_start[current_thread] = b + 1;
        }
    }

    // L'ultimo thread prende tutti i blocchi rimanenti
    thread_block_end[num_threads - 1] = matrix->num_blocks;

    // Stampa il numero di non-zero per ogni thread
    for (int t = 0; t < num_threads; t++) {
        long long thread_nnz = 0;
        for (int b = thread_block_start[t]; b < thread_block_end[t]; b++) {
            thread_nnz += nnz_per_block[b];
        }
    }

    free(nnz_per_block);

    return current_thread + 1; // Restituisce il numero effettivo di thread utilizzati
}

//Funzione per stampare i contenuti di una matrice HLL
void printHLLMatrix(HLLMatrix *hll) {
    if (hll->num_blocks < 3) {
        printf("HLL Matrix con %d blocchi:\n", hll->num_blocks);

        for (int b = 0; b < hll->num_blocks; b++) {
            ELLPACKBlock *block = &hll->blocks[b];
            printf("\nBlocco %d (%d righe, %d colonne, MAXNZ=%d):\n",
                   b, block->M, block->N, block->MAXNZ);

            for (int i = 0; i < block->M; i++) {
                printf("Riga %d: ", i);

                // Iteriamo attraverso i non-zeri in questa riga
                for (int j = 0; j < block->MAXNZ; j++) {
                    // Calcoliamo l'indice corretto nell'array linearizzato con layout trasposto
                    int idx = j * block->M + i;
                    if (block->JA[idx] >= 0 && block->JA[idx] < block->N) {
                        printf("(%d, %.6f) ", block->JA[idx], block->AS[idx]);
                    } else {
                        printf("(padding, %.6f) ", block->AS[idx]); // o altro messaggio per indicare padding
                    }
                }
                printf("\n");
            }
        }
    }
}







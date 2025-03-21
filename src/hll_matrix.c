#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "hll_matrix.h"
#include <immintrin.h>
#include <unistd.h>

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

            // Inizializzazione degli array con valori di default
            for (int i = 0; i < max_nnz * num_rows; i++) {
                hll->blocks[b].JA[i] = 0; // Inizializziamo a 0 invece di -1
                hll->blocks[b].AS[i] = 0.0;
            }
        } else {
            // Blocco vuoto
            hll->blocks[b].JA = NULL;
            hll->blocks[b].AS = NULL;
        }
    }

    // Array per tenere traccia dell'ultimo indice di colonna valido per ogni riga
    int *last_valid_col = (int *)calloc(M, sizeof(int));
    if (!last_valid_col) {
        printf("Errore di allocazione memoria per last_valid_col\n");
        free_hll_matrix(hll);
        free(row_counts);
        return -1;
    }

    // Inizializzazione a 0
    for (int i = 0; i < M; i++) {
        last_valid_col[i] = 0; // Valore di default per colonne quando non ci sono elementi
    }

    // Reset contatori per riempimento
    int *current_pos = calloc(M, sizeof(int));
    if (!current_pos) {
        printf("Errore di allocazione memoria per current_pos\n");
        free_hll_matrix(hll);
        free(row_counts);
        free(last_valid_col);
        return -1;
    }

    // Struttura per ordinare gli elementi per riga e colonna
    typedef struct {
        int col;
        double val;
    } ColValPair;

    // Array di array per memorizzare temporaneamente gli elementi ordinati per riga
    ColValPair **sorted_elements = malloc(M * sizeof(ColValPair *));
    if (!sorted_elements) {
        printf("Errore di allocazione memoria per sorted_elements\n");
        free_hll_matrix(hll);
        free(row_counts);
        free(last_valid_col);
        free(current_pos);
        return -1;
    }

    for (int i = 0; i < M; i++) {
        if (row_counts[i] > 0) {
            sorted_elements[i] = (ColValPair *)malloc(row_counts[i] * sizeof(ColValPair));
            if (!sorted_elements[i]) {
                printf("Errore di allocazione memoria per sorted_elements[%d]\n", i);
                // Pulizia memoria già allocata
                for (int j = 0; j < i; j++) {
                    if (sorted_elements[j]) free(sorted_elements[j]);
                }
                free(sorted_elements);
                free_hll_matrix(hll);
                free(row_counts);
                free(last_valid_col);
                free(current_pos);
                return -1;
            }
        } else {
            sorted_elements[i] = NULL;
        }
    }

    // Popola gli array temporanei
    for (int i = 0; i < pre->nz; i++) {
        const int row = pre->I[i];
        const int col = pre->J[i];
        const double val = pre->val[i];

        // Verifica validità indici
        if (row < 0 || row >= M || col < 0 || col >= pre->N) {
            printf("AVVISO: Indici non validi: riga=%d, colonna=%d\n", row, col);
            continue;
        }

        sorted_elements[row][current_pos[row]].col = col;
        sorted_elements[row][current_pos[row]].val = val;
        current_pos[row]++;
    }

    // Reset dei contatori
    memset(current_pos, 0, M * sizeof(int));

    // Ordina gli elementi per ogni riga per indice di colonna
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > 0) {
            // Ordinamento per indice di colonna (insertion sort)
            for (int j = 1; j < row_counts[i]; j++) {
                const ColValPair temp = sorted_elements[i][j];
                int k = j;
                while (k > 0 && sorted_elements[i][k-1].col > temp.col) {
                    sorted_elements[i][k] = sorted_elements[i][k-1];
                    k--;
                }
                sorted_elements[i][k] = temp;
            }
        }
    }

    // Riempimento degli array HLL con gli elementi ordinati
    for (int i = 0; i < M; i++) {
        const int block_idx = i / HACK_SIZE;
        const int local_row = i % HACK_SIZE;

        // Verifica validità blocco
        if (block_idx >= num_blocks) {
            printf("ERRORE: Blocco non valido: %d (max: %d)\n", block_idx, num_blocks-1);
            continue;
        }

        // Riempimento con i valori ordinati
        for (int j = 0; j < row_counts[i]; j++) {
            const int col = sorted_elements[i][j].col;
            const double val = sorted_elements[i][j].val;

            // Memorizza l'ultimo indice di colonna valido
            last_valid_col[i] = col;

            // Inserimento valore con layout trasposto: j * num_rows + local_row
            const int idx = j * hll->blocks[block_idx].M + local_row;

            hll->blocks[block_idx].JA[idx] = col;
            hll->blocks[block_idx].AS[idx] = val;
        }

        // Riempimento del padding con l'ultimo indice di colonna valido
        for (int j = row_counts[i]; j < hll->blocks[block_idx].MAXNZ; j++) {
            const int idx = j * hll->blocks[block_idx].M + local_row;
            hll->blocks[block_idx].JA[idx] = last_valid_col[i];
            hll->blocks[block_idx].AS[idx] = 0.0;
        }
    }

    // Calcolo memoria utilizzata
    size_t total_memory = 0;
    double max_block_mem = 0.0;

    for (int b = 0; b < num_blocks; b++) {
        if (hll->blocks[b].MAXNZ > 0) {
            const size_t block_mem = (size_t)hll->blocks[b].M * hll->blocks[b].MAXNZ * (sizeof(int) + sizeof(double));
            total_memory += block_mem;

            const double block_mem_gb = (double)block_mem / (1024.0 * 1024.0);
            if (block_mem_gb > max_block_mem) {
                max_block_mem = block_mem_gb;
            }
        }
    }

    const double total_mem_gb = (double)total_memory / (1024.0 * 1024.0);
    printf("Memoria HLL totale: %.5f MB, Memoria massima per blocco: %.5f MB\n",
           total_mem_gb, max_block_mem);

    // Pulizia memoria
    for (int i = 0; i < M; i++) {
        if (sorted_elements[i]) free(sorted_elements[i]);
    }
    free(sorted_elements);
    free(row_counts);
    free(current_pos);
    free(last_valid_col);
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

void spmv_hll_simd(const ELLPACKBlock *blocks, const double *x, double *y, int num_threads,
                   int const *thread_block_start, int const *thread_block_end) {

    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int start_block = thread_block_start[thread_id];
        const int end_block = thread_block_end[thread_id];

        // Calcola la moltiplicazione SpMV per i blocchi assegnati a questo thread
        for (int b = start_block; b < end_block; b++) {
            const ELLPACKBlock *block = &blocks[b];
            const int row_offset = b * HACK_SIZE;
            const int num_rows = block->M;
            const int maxnz = block->MAXNZ;
            const double *block_AS = block->AS;  // Cache locale dei puntatori
            const int *block_JA = block->JA;
            const int block_N = block->N;

            for (int i = 0; i < num_rows; i++) {
                double sum = 0.0;

                // Direttiva SIMD con allineamento e clausole aggiuntive
                #pragma omp simd reduction(+:sum)
                for (int j = 0; j < maxnz; j++) {
                    const int idx = j * num_rows + i;
                    const int col = block_JA[idx];
                    if (col < block_N) {
                        sum += block_AS[idx] * x[col];
                    }
                }

                y[row_offset + i] = sum;
            }
        }
    }
}

// Funzione serializzata per la moltiplicazione SpMV
void spmv_hll_serial(const int num_blocks, const ELLPACKBlock *blocks, const double *x, double *y) {
    // Elabora ogni blocco della matrice HLL
    for (int b = 0; b < num_blocks; b++) {
        const ELLPACKBlock *block = &blocks[b];  // Modificato qui: puntatore invece di copia
        const int row_offset = b * HACK_SIZE;    // Calcolo dell'offset delle righe globali
        const int num_rows = block->M;           // Modificato qui: -> invece di .

        // Per ogni riga nel blocco corrente
        for (int i = 0; i < num_rows; i++) {
            double sum = 0.0;

            // Per ogni possibile elemento non-zero nella riga
            for (int j = 0; j < block->MAXNZ; j++) {  // Modificato qui: -> invece di .
                // Accesso all'elemento (i,j) nel layout trasposto
                const int idx = j * num_rows + i;
                const int col = block->JA[idx];       // Modificato qui: -> invece di .

                // Verifica che l'indice di colonna sia valido
                if (col >= 0 && col < block->N) {      // Modificato qui: -> invece di .
                    sum += block->AS[idx] * x[col];    // Modificato qui: -> invece di .
                }
            }

            // Inserisce il risultato nella posizione corretta del vettore y
            y[row_offset + i] = sum;
        }
    }
}

void spmv_hll(const ELLPACKBlock *blocks, const double *x, double *y, int num_threads,
              int const *thread_block_start, int const *thread_block_end) {

    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int start_block = thread_block_start[thread_id];
        const int end_block = thread_block_end[thread_id];

        // Calcola la moltiplicazione SpMV per i blocchi assegnati a questo thread
        for (int b = start_block; b < end_block; b++) {
            const ELLPACKBlock *block = &blocks[b];
            const int row_offset = b * HACK_SIZE;
            const int num_rows = block->M;
            const int maxnz = block->MAXNZ;
            const double *block_AS = block->AS;  // Cache locale dei puntatori
            const int *block_JA = block->JA;
            // const int block_N = block->N; // Non abbiamo più bisogno di questo

            for (int i = 0; i < num_rows; i++) {
                double sum = 0.0;

                for (int j = 0; j < maxnz; j++) {
                    const int idx = j * num_rows + i;
                    const int col = block_JA[idx];
                    sum += block_AS[idx] * x[col];
                    // }
                }

                y[row_offset + i] = sum;
            }
        }
    }
}

int prepare_thread_distribution_hll(const HLLMatrix *matrix, int num_threads,
                                   int **thread_block_start, int **thread_block_end) {
    // Verifica parametri e allocazione memoria
    if (!matrix || num_threads <= 0 || !thread_block_start || !thread_block_end) {
        printf("Errore: parametri non validi in prepare_thread_distribution_hll\n");
        return 0;
    }

    // Usa meno thread se ci sono meno blocchi che thread
    if (matrix->num_blocks < num_threads) {
        num_threads = matrix->num_blocks;
    }

    // Allocazione memoria
    *thread_block_start = malloc((size_t)num_threads * sizeof(int));
    *thread_block_end = malloc((size_t)num_threads * sizeof(int));
    int *thread_nnz = malloc((size_t)num_threads * sizeof(int));

    if (*thread_block_start == NULL || *thread_block_end == NULL || thread_nnz == NULL) {
        // Gestione errore allocazione
        if (*thread_block_start) free(*thread_block_start);
        if (*thread_block_end) free(*thread_block_end);
        if (thread_nnz) free(thread_nnz);
        printf("Errore: allocazione memoria fallita in prepare_thread_distribution_hll\n");
        return 0;
    }

    // Inizializzazione array
    for (int t = 0; t < num_threads; t++) {
        (*thread_block_start)[t] = -1;
        (*thread_block_end)[t] = -1;
        thread_nnz[t] = 0;
    }

    // Calcolo del numero totale di elementi non zero effettivi
    long long total_nnz = 0;
    int *nnz_per_block = malloc(matrix->num_blocks * sizeof(int));

    if (!nnz_per_block) {
        free(*thread_block_start);
        free(*thread_block_end);
        free(thread_nnz);
        printf("Errore: allocazione memoria fallita per nnz_per_block\n");
        return 0;
    }

    // Contiamo gli elementi non zero effettivi per ogni blocco
    for (int b = 0; b < matrix->num_blocks; b++) {
        const ELLPACKBlock *block = &matrix->blocks[b];
        int block_nnz = 0;
        int num_rows = block->M;

        for (int i = 0; i < num_rows; i++) {
            int count = 0;
            for (int j = 0; j < block->MAXNZ; j++) {
                // Accesso all'elemento (i,j) nel layout trasposto
                int idx = j * num_rows + i;
                if (block->JA && idx < num_rows * block->MAXNZ &&
                    block->JA[idx] >= 0 && block->JA[idx] < block->N) {
                    count++;
                }
            }
            block_nnz += count;
        }

        nnz_per_block[b] = block_nnz;
        total_nnz += block_nnz;
    }

    // Numero target di elementi non zero per thread (divisione con arrotondamento per eccesso)
    const long long nnz_per_thread = (total_nnz + num_threads - 1) / num_threads;

    // Distribuzione dei blocchi in base agli NNZ
    int current_thread = 0;
    long long current_nnz = 0;

    for (int b = 0; b < matrix->num_blocks; b++) {
        // Imposta il blocco di inizio per il thread corrente se non è ancora impostato
        if ((*thread_block_start)[current_thread] == -1) {
            (*thread_block_start)[current_thread] = b;
        }

        current_nnz += nnz_per_block[b];
        thread_nnz[current_thread] += nnz_per_block[b];

        // Se questo thread ha raggiunto il target e non è l'ultimo thread
        if (current_nnz >= nnz_per_thread && current_thread < num_threads - 1) {
            (*thread_block_end)[current_thread] = b + 1; // Fine è esclusiva
            current_thread++;
            current_nnz = 0;
        }
    }

    // L'ultimo thread attivo prende tutti i blocchi rimanenti
    if (current_thread < num_threads) {
        (*thread_block_end)[current_thread] = matrix->num_blocks;
        current_thread++;
    }

    // Determinare quanti thread sono effettivamente utilizzati
    int valid_threads = 0;
    for (int t = 0; t < num_threads; t++) {
        if ((*thread_block_start)[t] != -1 && (*thread_block_end)[t] != -1 && thread_nnz[t] > 0) {
            // Mantieni i dati dei thread validi compattando gli array
            if (valid_threads != t) {
                (*thread_block_start)[valid_threads] = (*thread_block_start)[t];
                (*thread_block_end)[valid_threads] = (*thread_block_end)[t];
                thread_nnz[valid_threads] = thread_nnz[t];
            }
            valid_threads++;
        }
    }

    // Aggiorna il numero effettivo di thread utilizzati
    num_threads = valid_threads;

    // Ottimizzazione: bilanciamento del carico tra thread adiacenti
    for (int t = 0; t < num_threads - 1; t++) {
        // Verifica se il thread successivo ha significativamente più lavoro (più di 1.5x)
        if (thread_nnz[t+1] > thread_nnz[t] * 1.5) {
            int start_block = (*thread_block_start)[t+1];
            int block_nnz = nnz_per_block[start_block];

            // Sposta solo il blocco se non sovraccarica il thread corrente
            if (thread_nnz[t] + block_nnz <= nnz_per_thread * 1.2) {
                // Sposta un blocco dal thread successivo al thread corrente
                (*thread_block_end)[t]++;
                (*thread_block_start)[t+1]++;

                // Aggiorna i conteggi NNZ
                thread_nnz[t] += block_nnz;
                thread_nnz[t+1] -= block_nnz;
            }
        }
    }

    // Stampe dettagliate per ogni thread
    printf("\n--- Dettagli distribuzione thread per HLL ---\n");
    printf("Thread attivi: %d (su %d richiesti inizialmente)\n", valid_threads, num_threads);
    printf("Numero totale di nnz: %lld\n", total_nnz);
    printf("Numero totale di blocchi: %d\n\n", matrix->num_blocks);

    for (int t = 0; t < num_threads; t++) {
        int num_blocks = (*thread_block_end)[t] - (*thread_block_start)[t];
        long long actual_nnz = 0;

        // Calcola il numero effettivo di nnz per questo thread
        for (int b = (*thread_block_start)[t]; b < (*thread_block_end)[t]; b++) {
            actual_nnz += nnz_per_block[b];
        }

        printf("Thread %d: %d blocchi (da %d a %d), %lld nnz (%.2f%% del totale)\n",
                t,
                num_blocks,
                (*thread_block_start)[t],
                (*thread_block_end)[t] - 1,
                actual_nnz,
                actual_nnz * 100.0 / total_nnz);
    }

    printf("--- Fine dettagli distribuzione HLL ---\n\n");

    free(nnz_per_block);
    free(thread_nnz);
    return valid_threads;
}

//Funzione per stampare i contenuti di una matrice HLL
void printHLLMatrix(HLLMatrix *hll) {
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
    sleep(3);
}







#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <unistd.h>
#include <omp.h>
#include <hll_matrix.h>
#include <matrix_parser.h>
#include <utility.h>

void init_hll_matrix(HLLMatrix *hll) {
    hll->num_blocks = 0;
    hll->blocks = NULL;
}

// Struttura per ordinare gli elementi per riga e colonna
typedef struct {
    int col;
    double val;
} ColValPair;

int compareColValPair(const void *a, const void *b) {
    // Converti i puntatori void* ai tipi corretti (ColValPair*)
    ColValPair *pairA = (ColValPair *)a;
    ColValPair *pairB = (ColValPair *)b;

    // Confronta in base al campo 'col'
    if (pairA->col < pairB->col) {
        return -1; // pairA viene prima di pairB
    } else if (pairA->col > pairB->col) {
        return 1;  // pairA viene dopo pairB
    } else {
        return 0;  // Le colonne sono uguali
    }
}

int convert_to_hll(const PreMatrix *pre, HLLMatrix *hll) {
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
    int *row_counts = (int *)calloc(M, sizeof(int));
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
            FREE_CHECK(row_counts);
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

        if (max_nnz > 0) {
            hll->blocks[b].JA = (int *)malloc(max_nnz * num_rows * sizeof(int));
            if (!hll->blocks[b].JA) {
                printf("ERRORE: Allocazione fallita per JA del blocco %d\n", b);
                free_hll_matrix(hll);
                FREE_CHECK(row_counts);
                return -1;
            }

            hll->blocks[b].AS = (double *)malloc(max_nnz * num_rows * sizeof(double));
            if (!hll->blocks[b].AS) {
                printf("ERRORE: Allocazione fallita per AS del blocco %d\n", b);
                free_hll_matrix(hll);
                FREE_CHECK(row_counts);
                return -1;
            }

            // Inizializzazione degli array con valori di default
            for (int i = 0; i < max_nnz * num_rows; i++) {
                hll->blocks[b].JA[i] = 0;
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
        FREE_CHECK(row_counts);
        return -1;
    }

    // Inizializzazione a 0
    for (int i = 0; i < M; i++) {
        last_valid_col[i] = 0; // Valore di default per colonne quando non ci sono elementi
    }

    // Reset contatori per riempimento
    int *current_pos = (int *)calloc(M, sizeof(int));
    if (!current_pos) {
        printf("Errore di allocazione memoria per current_pos\n");
        free_hll_matrix(hll);
        FREE_CHECK(row_counts);
        FREE_CHECK(last_valid_col);
        return -1;
    }

    // Array di array per memorizzare temporaneamente gli elementi ordinati per riga
    ColValPair **sorted_elements = (ColValPair **)malloc(M * sizeof(ColValPair *));
    if (!sorted_elements) {
        printf("Errore di allocazione memoria per sorted_elements\n");
        free_hll_matrix(hll);
        FREE_CHECK(row_counts);
        FREE_CHECK(last_valid_col);
        FREE_CHECK(current_pos);
        return -1;
    }

    for (int i = 0; i < M; i++) {
        if (row_counts[i] > 0) {
            sorted_elements[i] = (ColValPair *)malloc(row_counts[i] * sizeof(ColValPair));
            if (!sorted_elements[i]) {
                printf("Errore di allocazione memoria per sorted_elements[%d]\n", i);
                // Pulizia memoria già allocata
                for (int j = 0; j < i; j++) {
                    if (sorted_elements[j]) FREE_CHECK(sorted_elements[j]);
                }
                FREE_CHECK(sorted_elements);
                free_hll_matrix(hll);
                FREE_CHECK(row_counts);
                FREE_CHECK(last_valid_col);
                FREE_CHECK(current_pos);
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
    //memset(current_pos, 0, M * sizeof(int));

    for (int i = 0; i < M; i++) {
        if (row_counts[i] > 1) {
            qsort(
                sorted_elements[i],      // Puntatore all'inizio dell'array da ordinare (per la riga i)
                (size_t)row_counts[i],   // Numero di elementi nell'array per la riga i (cast a size_t)
                sizeof(ColValPair),      // Dimensione in byte di ciascun elemento (la struct)
                compareColValPair        // Puntatore alla funzione di confronto definita sopra
            );
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

            // Inserimento valore con layout trasposto
            const int idx = local_row * hll->blocks[block_idx].MAXNZ + j;

            hll->blocks[block_idx].JA[idx] = col;
            hll->blocks[block_idx].AS[idx] = val;
        }

        // Riempimento del padding con l'ultimo indice di colonna valido
        for (int j = row_counts[i]; j < hll->blocks[block_idx].MAXNZ; j++) {
            const int idx = local_row * hll->blocks[block_idx].MAXNZ + j;
            hll->blocks[block_idx].JA[idx] = last_valid_col[i];
            hll->blocks[block_idx].AS[idx] = 0.0;
        }
    }
    // Pulizia memoria
    for (int i = 0; i < M; i++) {
        if (sorted_elements[i]) FREE_CHECK(sorted_elements[i]);
    }
    FREE_CHECK(sorted_elements);
    FREE_CHECK(row_counts);
    FREE_CHECK(current_pos);
    FREE_CHECK(last_valid_col);
    return 0;
}

// Funzione per liberare la memoria di una matrice HLL
void free_hll_matrix(HLLMatrix *hll) {
    if (!hll) return;

    // Liberazione della memoria per ciascun blocco
    if (hll->blocks) {
        for (int b = 0; b < hll->num_blocks; b++) {
            if (hll->blocks[b].JA) {
                FREE_CHECK(hll->blocks[b].JA);
                hll->blocks[b].JA = NULL;
            }

            if (hll->blocks[b].AS) {
                FREE_CHECK(hll->blocks[b].AS);
                hll->blocks[b].AS = NULL;
            }
        }
        FREE_CHECK(hll->blocks);
        hll->blocks = NULL;
    }
    // Resetta i valori
    hll->num_blocks = 0;
}



// Funzione serializzata per la moltiplicazione SpMV
void spmv_hll_serial(const int num_blocks, const ELLPACKBlock *blocks, const double *x, double *y) {
    // Elabora ogni blocco della matrice HLL
    for (int b = 0; b < num_blocks; b++) {
        const ELLPACKBlock *block = &blocks[b];
        const int row_offset = b * HACK_SIZE;    // Calcolo dell'offset delle righe globali
        const int num_rows = block->M;

        // Per ogni riga nel blocco corrente
        for (int i = 0; i < num_rows; i++) {
            double sum = 0.0;

            // Per ogni possibile elemento non-zero nella riga
            for (int j = 0; j < block->MAXNZ; j++) {
                const int idx = i * block->MAXNZ + j;
                const int col = block->JA[idx];
                sum += block->AS[idx] * x[col];
            }

            // Inserisce il risultato nella posizione corretta del vettore y
            y[row_offset + i] = sum;
        }
    }
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
                    // Calcoliamo l'indice corretto nell'array
                    int idx = i * block->MAXNZ + j;
                    if (block->JA[idx] >= 0 && block->JA[idx] < block->N) {
                        printf("(%d, %.6f) ", block->JA[idx], block->AS[idx]);
                    } else {
                        printf("(padding, %.6f) ", block->AS[idx]);
                    }
                }
                printf("\n");
            }
        }
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
                    const int idx = i * maxnz + j;
                    const double val = block_AS[idx];
                        const int col = block_JA[idx];
                        sum += val * x[col];
                }

                y[row_offset + i] = sum;
            }
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

            for (int i = 0; i < num_rows; i++) {
                double sum = 0.0;

                for (int j = 0; j < maxnz; j++) {
                    const int idx = i * maxnz + j;
                    const double val = block_AS[idx];
                        const int col = block_JA[idx];
                        sum += val * x[col];
                }

                y[row_offset + i] = sum;
            }
        }
    }
}

int prepare_thread_distribution_hll(const HLLMatrix *matrix, int num_threads,
                                   int **thread_block_start, int **thread_block_end) {
    // Verifica parametri
    if (!matrix || num_threads <= 0 || !thread_block_start || !thread_block_end) {
        printf("Errore: parametri non validi in prepare_thread_distribution_hll\n");
        return 0;
    }

    // Limita il numero di thread se ci sono meno blocchi che thread
    num_threads = (matrix->num_blocks < num_threads) ? matrix->num_blocks : num_threads;

    // Allocazione memoria
    *thread_block_start = malloc((size_t)num_threads * sizeof(int));
    *thread_block_end = malloc((size_t)num_threads * sizeof(int));
    int *thread_nnz = malloc((size_t)num_threads * sizeof(int));
    int *nnz_per_block = malloc(matrix->num_blocks * sizeof(int));

    if (!*thread_block_start || !*thread_block_end || !thread_nnz || !nnz_per_block) {
        // Gestione errore allocazione
        FREE_CHECK(*thread_block_start);
        FREE_CHECK(*thread_block_end);
        FREE_CHECK(thread_nnz);
        FREE_CHECK(nnz_per_block);
        printf("Errore: allocazione memoria fallita\n");
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

    // Numero target di elementi non zero per thread
    const long long nnz_per_thread = (total_nnz + num_threads - 1) / num_threads;

    // Distribuzione dei blocchi in base agli NNZ
    int current_thread = 0;
    long long current_nnz = 0;

    for (int b = 0; b < matrix->num_blocks; b++) {
        // Imposta il blocco di inizio per il thread corrente
        if ((*thread_block_start)[current_thread] == -1) {
            (*thread_block_start)[current_thread] = b;
        }

        current_nnz += nnz_per_block[b];
        thread_nnz[current_thread] += nnz_per_block[b];

        // Se questo thread ha raggiunto il target e non è l'ultimo thread
        if (current_nnz >= nnz_per_thread && current_thread < num_threads - 1) {
            (*thread_block_end)[current_thread] = b + 1;
            current_thread++;
            current_nnz = 0;
        }
    }

    // L'ultimo thread attivo prende tutti i blocchi rimanenti
    if (current_thread < num_threads) {
        (*thread_block_end)[current_thread] = matrix->num_blocks;
        current_thread++;
    }

    // Compattazione degli array
    int valid_threads = 0;
    for (int t = 0; t < num_threads; t++) {
        if ((*thread_block_start)[t] != -1 && (*thread_block_end)[t] != -1 && thread_nnz[t] > 0) {
            if (valid_threads != t) {
                (*thread_block_start)[valid_threads] = (*thread_block_start)[t];
                (*thread_block_end)[valid_threads] = (*thread_block_end)[t];
                thread_nnz[valid_threads] = thread_nnz[t];
            }
            valid_threads++;
        }
    }

    // Stampe dettagliate per ogni thread
    printf("\n--- Dettagli distribuzione thread per HLL ---\n");
    printf("Thread attivi: %d (su %d richiesti inizialmente)\n", valid_threads, num_threads);
    printf("Numero totale di nnz: %lld\n", total_nnz);
    printf("Numero totale di blocchi: %d\n\n", matrix->num_blocks);

    for (int t = 0; t < valid_threads; t++) {
        int num_blocks = (*thread_block_end)[t] - (*thread_block_start)[t];
        long long actual_nnz = 0;

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







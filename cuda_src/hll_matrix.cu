#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <unistd.h>
#include <hll_matrix.cuh>

void init_hll_matrix(HLLMatrix *hll) {
    hll->num_blocks = 0;
    hll->blocks = NULL;
}

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

        // Allocazione degli array JA e AS in formato trasposto
        if (max_nnz > 0) {
            // Allocazione linearizzata di JA (MAXNZ x num_rows)
            hll->blocks[b].JA = (int *)malloc(max_nnz * num_rows * sizeof(int));
            if (!hll->blocks[b].JA) {
                printf("ERRORE: Allocazione fallita per JA del blocco %d\n", b);
                free_hll_matrix(hll);
                FREE_CHECK(row_counts);
                return -1;
            }

            // Allocazione linearizzata di AS (MAXNZ x num_rows)
            hll->blocks[b].AS = (double *)malloc(max_nnz * num_rows * sizeof(double));
            if (!hll->blocks[b].AS) {
                printf("ERRORE: Allocazione fallita per AS del blocco %d\n", b);
                free_hll_matrix(hll);
                FREE_CHECK(row_counts);
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

    // Ordina gli elementi per ogni riga per indice di colonna
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
        if (sorted_elements[i]) free(sorted_elements[i]);
    }
    FREE_CHECK(sorted_elements);
    FREE_CHECK(row_counts);
    FREE_CHECK(current_pos);
    FREE_CHECK(last_valid_col);
    return 0;
}

int compute_max_MAXNZ(const ELLPACKBlock* blocks, int num_blocks) {
    int max_MAXNZ = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (blocks[i].MAXNZ > max_MAXNZ) {
            max_MAXNZ = blocks[i].MAXNZ;
        }
    }
    return max_MAXNZ;
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

// Funzione per stampare i contenuti di una matrice HLL
void printHLLMatrix(HLLMatrix *hll) {
    printf("HLL Matrix con %d blocchi:\n", hll->num_blocks);

    for (int b = 0; b < 2; b++) {
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

__global__ void spmv_hll_naive_kernel(
    int total_M,
    const ELLPACKBlock *blocks, // Puntatore all'array di blocchi su GPU
    const double *x,
    double *y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ogni thread calcola un elemento y[tid], se tid è una riga valida
    if (tid < total_M) {

        int block_idx = tid / HACK_SIZE; //tid è la riga, tid / HACK_SIZE è il blocco
        int local_row = tid % HACK_SIZE; //Ora siamo nel blocco, e quindi uso operatore %


        ELLPACKBlock block = blocks[block_idx];

        const int* block_JA    = block.JA;
        const double* block_AS    = block.AS;
        int           block_MAXNZ = block.MAXNZ;

        double sum = 0.0;

        for (int j = 0; j < block_MAXNZ; j++) {
            int index = local_row * block_MAXNZ + j;
            int col = block_JA[index];
            double val = block_AS[index];
            sum += val * x[col];
        }
        y[tid] = sum;
    }
}

__global__ void spmv_hll_warp_kernel(
    int total_M,
    const ELLPACKBlock *blocks, // Puntatore all'array di blocchi su GPU
    const double *x,
    double *y) {
    // Calcola l'indice globale della riga che questo warp gestirà
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    // Calcola l'ID del thread all'interno del warp (0-31)
    int lane_id = threadIdx.x % 32;

    // Verifica se il warp_id corrisponde a una riga valida della matrice
    if (warp_id < total_M) {
        // Calcola l'indice del blocco e la riga locale dentro quel blocco
        int block_idx = warp_id / HACK_SIZE;
        int local_row = warp_id % HACK_SIZE; // Riga da usare dentro il blocco

        // Accedi al blocco corretto
        ELLPACKBlock block = blocks[block_idx];

        // Ottieni i puntatori e le dimensioni del blocco specifico
        const int* block_JA    = block.JA;
        const double* block_AS = block.AS;
        int           block_MAXNZ = block.MAXNZ; // Max non-zeri per riga nel blocco

        double sum = 0.0;

        // Il loop avanza di 32 (warp size) ad ogni iterazione
        for (int j = lane_id; j < block_MAXNZ; j += 32) {
            // Calcola l'indice lineare usando la riga LOCALE
            int index = local_row * block_MAXNZ + j;

            int col = block_JA[index];
            double val = block_AS[index];

            sum += val * x[col];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        }

        if (lane_id == 0) {
            y[warp_id] = sum;
        }
    }
}

__global__ void spmv_hll_warp_shared_kernel_v1(
    int total_M,
    const ELLPACKBlock *blocks,
    const double *x,
    double *y)
{
    // Shared memory per i valori di x usati da ciascun warp
    extern __shared__ double s_x[];

    // Calcola gli indici
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;

    // Esci se fuori dai limiti
    if (warp_id >= total_M) return;

    // Calcola l'indice del blocco e la riga locale
    int block_idx = warp_id / HACK_SIZE;
    int local_row = warp_id % HACK_SIZE;

    // Accedi al blocco
    ELLPACKBlock block = blocks[block_idx];
    int block_MAXNZ = block.MAXNZ;

    // Puntatore alla shared memory per questo warp
    double* warp_x = &s_x[warp_in_block * block_MAXNZ];

    // Carica in shared memory i valori x necessari
    for (int j = lane_id; j < block_MAXNZ; j += 32) {
        int col = block.JA[local_row * block_MAXNZ + j]; //Prendo il valore della colonna, i thread del warp collaborano
        warp_x[j] = x[col];
    }

    __syncwarp();

    // Calcola la somma
    double sum = 0.0;
    for (int j = lane_id; j < block_MAXNZ; j += 32) {
        sum += block.AS[local_row * block_MAXNZ + j] * warp_x[j];
    }

    // Riduzione warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Thread 0 scrive il risultato
    if (lane_id == 0) {
        y[warp_id] = sum;
    }
}

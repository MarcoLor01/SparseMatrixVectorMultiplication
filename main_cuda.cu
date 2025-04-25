#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include "performance_calculate.cuh"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include "csr_matrix_cuda.cuh"
#include "matrix_parser.cuh"
#include "utility.cuh"
#include "hll_matrix.cuh"

#define NUM_ITERATION 100

int process_matrix_file(const char *filepath, PreMatrix *pre_mat) {
    init_pre_matrix(pre_mat);

    printf("\n===========================================\n");
    printf("Elaborazione matrice: %s\n", filepath);
    printf("===========================================\n");

    if (read_matrix_market(filepath, pre_mat) != 0) {
        printf("Errore nella lettura della matrice\n");
        return -1;
    }
    return 0;
}

int clear_directory(const char *path) {
    DIR *dir = opendir(path);
    if (dir == NULL) {
        perror("Errore nell'aprire la directory");
        return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // Evita di rimuovere "." e ".."
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", path, entry->d_name);

            if (remove(file_path) == -1) {
                perror("Errore nell'eliminare il file");
                closedir(dir);
                return -1;
            }
        }
    }

    closedir(dir);
    return 0;
}



int main() {

    struct dirent *dir;
    const char *directory_path = "../matrix_for_test/";
    //const char *directory_path = "../matrix_generated/";
	const char *output_csv = "../result/spmv_results_cuda.csv";

    if (mkdir("../result", 0777) == -1) {
        if (errno == EEXIST) {
            printf("La directory 'result' esiste già. Svuotiamo il suo contenuto...\n");
            if (clear_directory("../result") == -1) {
                printf("Errore nel svuotare la directory.\n");
                exit(EXIT_FAILURE);
            }
            printf("La directory è stata svuotata.\n");
        } else {
            perror("Errore nella creazione della directory");
            exit(EXIT_FAILURE);
        }
    } else {
        printf("Directory 'result' creata con successo.\n");
    }

    // Apertura directory contenente le matrici
    DIR *d = opendir(directory_path);
    if (d == NULL) {
        printf("Errore nell'apertura della directory %s\n", directory_path);
        exit(EXIT_FAILURE);
    }

    // Per ogni matrice nella directory
    while ((dir = readdir(d)) != NULL) {
        // Salta file che non sono matrici (.mtx)
        if (dir->d_name[0] == '.' || strlen(dir->d_name) < 4 ||
            strcmp(dir->d_name + strlen(dir->d_name) - 4, ".mtx") != 0) {
            continue;
            }

        // Preparazione percorso completo e lettura matrice
        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s%s", directory_path, dir->d_name);
        printf("\nCaricamento matrice: %s\n", dir->d_name);

        // Inizializzazione e caricamento matrice in formato pre-elaborazione
        PreMatrix pre_mat;
        init_pre_matrix(&pre_mat);
        if (process_matrix_file(full_path, &pre_mat) != 0) {
            fprintf(stderr, "Errore nel processamento della matrice\n");
            continue;
        }

        // Conversione in formato CSR
        CSRMatrix csr_mat;
        if (convert_in_csr(&pre_mat, &csr_mat) != 0) {
            fprintf(stderr, "Errore nella conversione in CSR\n");
            free_pre_matrix(&pre_mat);
            continue;
        }

        // Conversione in formato HLL/ELLPACK
        HLLMatrix hll_mat;
        if (convert_to_hll(&pre_mat, &hll_mat) != 0) {
            fprintf(stderr, "Errore nella conversione in ELLPACK");
            free_csr_matrix(&csr_mat);
            free_pre_matrix(&pre_mat);
            continue;
        }

        // Inizializzazione vettore x
        double *x = (double *)malloc(csr_mat.N * sizeof(double));
		init_vector_at_one(x, csr_mat.N);

        double *y_serial = (double *)malloc(csr_mat.M * sizeof(double));
        double *y_parallel_csr = (double *)malloc(csr_mat.M * sizeof(double));
        double *y_hll = (double *)malloc(csr_mat.M * sizeof(double));

        // ================================
        // TEST 1: VERSIONE SERIALE (CSR)
        // ================================
        cudaEvent_t start, stop;
    	CUDA_CHECK(cudaEventCreate(&start));
    	CUDA_CHECK(cudaEventCreate(&stop));

        // Esecuzione seriale per confronto
    	printf("\n--- TEST SERIALE CSR ---\n");
    	reset_medium_time_metrics();
        for(int i = 0; i < NUM_ITERATION; i++){
            printf("Iterazione numero: %d\n", i+1);
            memset(y_serial, 0, csr_mat.M * sizeof(double));

            cudaEventRecord(start);
        	csr_matrix_vector_mult(csr_mat.M, csr_mat.row_ptr, csr_mat.col_idx,
            	                       csr_mat.values, x, y_serial);
            cudaEventRecord(stop);

    		// Aspetta la fine dell'esecuzione
    		cudaEventSynchronize(stop);

    		// Calcola il tempo trascorso
    		float milliseconds = 0;
    		cudaEventElapsedTime(&milliseconds, start, stop);
        	update_medium_metric(SERIAL_TIME, milliseconds / 1000.0f);

            double flops_serial = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
			printf("Tempo impiegato per l'esecuzione seriale CSR: %f secondi\n", milliseconds/1000.0f);
			printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_serial);
            print_flops(flops_serial);
		}

        double avg_time_seconds_serial = get_metric_value(SERIAL_TIME) / NUM_ITERATION;
        double avg_flops_serial = calculate_flops(csr_mat.nz, avg_time_seconds_serial);
    	printf("Tempo medio impiegato per l'esecuzione seriale CSR: %f secondi\n", avg_time_seconds_serial);

    	// Dichiarazione dei puntatori per i dati sulla GPU
    	int *d_row_ptr;
    	int *d_col_idx;
    	double *d_values;
    	double *d_x;
    	double *d_y;

    	// Allocazione memoria sulla GPU
    	CUDA_CHECK(cudaMalloc((void **)&d_row_ptr, (csr_mat.M + 1) * sizeof(int)));
    	CUDA_CHECK(cudaMalloc((void **)&d_col_idx, csr_mat.nz * sizeof(int)));
    	CUDA_CHECK(cudaMalloc((void **)&d_values, csr_mat.nz * sizeof(double)));
    	CUDA_CHECK(cudaMalloc((void **)&d_x, csr_mat.N * sizeof(double)));
    	CUDA_CHECK(cudaMalloc((void **)&d_y, csr_mat.M * sizeof(double)));

    	// Copia dei dati dalla CPU alla GPU (una sola volta)
    	CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (csr_mat.M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    	CUDA_CHECK(cudaMemcpy(d_col_idx, csr_mat.col_idx, csr_mat.nz * sizeof(int), cudaMemcpyHostToDevice));
    	CUDA_CHECK(cudaMemcpy(d_values, csr_mat.values, csr_mat.nz * sizeof(double), cudaMemcpyHostToDevice));
    	CUDA_CHECK(cudaMemcpy(d_x, x, csr_mat.N * sizeof(double), cudaMemcpyHostToDevice));

        // Configurazione dei parametri per il lancio del kernel
        int minGrid, blockSize, blocksPerGrid, threadsPerBlock;
		cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, spmv_csr_naive_kernel, 0, csr_mat.M);
		blocksPerGrid   = (csr_mat.M + blockSize - 1) / blockSize;
		threadsPerBlock = blockSize;

        // ======================================
        // TEST 2: VERSIONE THREAD PER RIGA (CSR)
        // ======================================
    	printf("\n--- TEST THREAD PER RIGA CSR ---\n");
    	for(int i = 0; i < NUM_ITERATION; i++) {
            printf("Iterazione numero: %d\n", i);
            CUDA_CHECK(cudaMemset(d_y, 0, csr_mat.M * sizeof(double)));
            // Registra l'evento di inizio
            CUDA_CHECK(cudaEventRecord(start));

            // Lancio del kernel CUDA
            spmv_csr_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                csr_mat.M, d_row_ptr, d_col_idx, d_values, d_x, d_y
            );

            CUDA_CHECK(cudaGetLastError());

            // Registra l'evento di fine
            CUDA_CHECK(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            CUDA_CHECK(cudaEventSynchronize(stop));

            // Calcola il tempo trascorso
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            update_medium_metric(ROW_CSR_TIME, milliseconds / 1000.0f);


            printf("Tempo impiegato per l'esecuzione del Kernel-Naive: %f secondi\n", milliseconds/1000.0f);
            double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
            printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
            print_flops(flops_parallel);

            // Copia il risultato ad ogni iterazione
            CUDA_CHECK(cudaMemcpy(y_parallel_csr, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

            // Controlla la correttezza del risultato ad ogni iterazione
            printf("Verifica risultato iterazione %d:\n", i);
            bool error_csr = checkDifferences(y_serial, y_parallel_csr, csr_mat.M, false);

	    }
    	// Calcola e stampa il tempo medio in secondi
    	double avg_time_seconds_naive = get_metric_value(ROW_CSR_TIME) / NUM_ITERATION;
        double avg_flops_parallel = calculate_flops(csr_mat.nz, avg_time_seconds_naive);
    	printf("Tempo medio impiegato per l'esecuzione del Kernel-Naive CSR: %f secondi\n", avg_time_seconds_naive);
		printf("FLOPS medi: ");
        print_flops(avg_flops_parallel);

        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                  spmv_csr_warp_kernel, 0, 0);

		// Assicurati che blockSize sia multiplo di 32 (dimensione di un warp)
		blockSize = (blockSize / 32) * 32;
		int warps_per_block = blockSize / 32;
		int threadsPerBlock_warp = blockSize;

		// Un warp per riga, quindi M warps totali
		int blocksPerGrid_warp = (csr_mat.M + warps_per_block - 1) / warps_per_block;

        // ======================================
        // TEST 3: VERSIONE WARP PER RIGA (CSR)
        // ======================================
    	printf("\n--- TEST WARP PER RIGA CSR ---\n");
    	clear_gpu_cache(64);
        for(int i = 0; i < NUM_ITERATION; i++) {
            printf("Iterazione numero: %d\n", i);
            CUDA_CHECK(cudaMemset(d_y, 0, csr_mat.M * sizeof(double)));
            // Registra l'evento di inizio
            CUDA_CHECK(cudaEventRecord(start));

            // Lancio del kernel CUDA warp-based
            spmv_csr_warp_kernel<<<blocksPerGrid_warp, threadsPerBlock_warp>>>(
                csr_mat.M, d_row_ptr, d_col_idx, d_values, d_x, d_y
            );

            CUDA_CHECK(cudaGetLastError());

            // Registra l'evento di fine
            CUDA_CHECK(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            CUDA_CHECK(cudaEventSynchronize(stop));

            // Calcola il tempo trascorso
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        	update_medium_metric(WARP_CSR_TIME, milliseconds / 1000.0f);

            printf("Tempo impiegato per l'esecuzione del Kernel-Warp: %f secondi\n", milliseconds/1000.0f);
            double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
            printf("Tempo esecuzione parallelo Warp CSR: %f secondi\n", milliseconds/1000.0f);
            printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
            print_flops(flops_parallel);

            // Copia il risultato ad ogni iterazione
            CUDA_CHECK(cudaMemcpy(y_parallel_csr, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

            // Controlla la correttezza del risultato ad ogni iterazione
            printf("Verifica risultato iterazione %d:\n", i);
            bool error_warp = checkDifferences(y_serial, y_parallel_csr, csr_mat.M, false);

        }
        // Calcola e stampa il tempo medio in secondi
        double avg_time_seconds_warp = get_metric_value(WARP_CSR_TIME) / NUM_ITERATION;
        printf("Tempo medio impiegato per l'esecuzione del Kernel-Warp CSR: %f secondi\n", avg_time_seconds_warp);
		double avg_flops_warp = calculate_flops(csr_mat.nz, avg_time_seconds_warp);
        printf("Flops medi: ");
        print_flops(avg_flops_warp);

        // ==================================================
        // TEST 4: VERSIONE WARP PER RIGA E MEMORIA CONDIVISA (CSR)
        // ==================================================
		size_t sharedMemSize = sizeof(double) * min(csr_mat.N, MAX_CACHE);

		// Funzione lambda per calcolare la shared memory in base alla dimensione del blocco
		auto sharedMemCalculator = [&](int blockSize) {
    		return sharedMemSize;
		};

		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGrid, &blockSize,
                                             spmv_csr_warp_shared_memory_kernel,
                                             sharedMemCalculator, 0);

		blockSize = (blockSize / 32) * 32;
		int warpsPerBlock = blockSize / 32;
		threadsPerBlock = blockSize;

		// Un warp per riga, quindi M warps totali
		blocksPerGrid = (csr_mat.M + warpsPerBlock - 1) / warpsPerBlock;
        printf("\n--- TEST WARP PER RIGA CON MEMORIA CONDIVISA E CSR ---\n");
		clear_gpu_cache(64);
        for(int i = 0; i < NUM_ITERATION; i++) {
            printf("Iterazione numero: %d\n", i);
            CUDA_CHECK(cudaMemset(d_y, 0, csr_mat.M * sizeof(double)));
            // Registra l'evento di inizio
            CUDA_CHECK(cudaEventRecord(start));

            // Lancio del kernel CUDA warp-based con shared memory
            spmv_csr_warp_shared_memory_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                csr_mat.M, d_row_ptr, d_col_idx, d_values, d_x, d_y
            );

            CUDA_CHECK(cudaGetLastError());

            // Registra l'evento di fine
            CUDA_CHECK(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            CUDA_CHECK(cudaEventSynchronize(stop));

            // Calcola il tempo trascorso
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        	update_medium_metric(WARP_SHARED_MEMORY_CSR_TIME, milliseconds / 1000.0f);

            printf("Tempo impiegato per l'esecuzione del Kernel-Warp com shared memory CSR: %f secondi\n", milliseconds/1000.0f);
            double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
            printf("Tempo esecuzione parallelo Warp shared memory CSR: %f secondi\n", milliseconds/1000.0f);
            printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
            print_flops(flops_parallel);

            // Copia il risultato ad ogni iterazione
            CUDA_CHECK(cudaMemcpy(y_parallel_csr, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

            // Controlla la correttezza del risultato ad ogni iterazione
            printf("Verifica risultato iterazione %d:\n", i);
            bool error_warp = checkDifferences(y_serial, y_parallel_csr, csr_mat.M, false);

        }
        // Calcola e stampa il tempo medio in secondi
        double avg_time_seconds_warp_shared = get_metric_value(WARP_SHARED_MEMORY_CSR_TIME) / NUM_ITERATION;
        printf("Tempo medio impiegato per l'esecuzione del Kernel-Warp shared memory CSR: %f secondi\n", avg_time_seconds_warp_shared);
		double avg_flops_warp_shared = calculate_flops(csr_mat.nz, avg_time_seconds_warp_shared);
        printf("Flops medi: ");
        print_flops(avg_flops_warp_shared);

    	// Vettori x e y sulla device
    	double *d_x_hll;
    	double *d_y_hll;
    	CUDA_CHECK(cudaMalloc((void **)&d_x_hll, csr_mat.N * sizeof(double)));
    	CUDA_CHECK(cudaMalloc((void **)&d_y_hll, csr_mat.M * sizeof(double)));
    	// Copia dei dati x dalla CPU alla GPU
    	CUDA_CHECK(cudaMemcpy(d_x_hll, x, csr_mat.N * sizeof(double), cudaMemcpyHostToDevice));
    	// Alloca array di blocchi ELLPACK sulla GPU
    	ELLPACKBlock *d_blocks;
    	cudaMalloc((void **)&d_blocks, hll_mat.num_blocks * sizeof(ELLPACKBlock));
    	// Crea array temporaneo di blocchi sull'host
    	ELLPACKBlock *temp_blocks = (ELLPACKBlock *)malloc(hll_mat.num_blocks * sizeof(ELLPACKBlock));
    	// Alloca e copia ogni blocco ELLPACK
    	for (int b = 0; b < hll_mat.num_blocks; b++) {
    		ELLPACKBlock h_block = hll_mat.blocks[b];

    		// Allocazione memoria per i dati del blocco (layout trasposto)
    		int *d_JA;
    		double *d_AS;
    		CUDA_CHECK(cudaMalloc((void **)&d_JA, h_block.MAXNZ * h_block.M * sizeof(int)));
    		CUDA_CHECK(cudaMalloc((void **)&d_AS, h_block.MAXNZ * h_block.M * sizeof(double)));

    		// Copia dei dati del blocco
    		CUDA_CHECK(cudaMemcpy(d_JA, h_block.JA, h_block.MAXNZ * h_block.M * sizeof(int), cudaMemcpyHostToDevice));
    		CUDA_CHECK(cudaMemcpy(d_AS, h_block.AS, h_block.MAXNZ * h_block.M * sizeof(double), cudaMemcpyHostToDevice));

    		// Imposta i campi del blocco temporaneo
    		temp_blocks[b].M = h_block.M;
    		temp_blocks[b].N = h_block.N;
    		temp_blocks[b].MAXNZ = h_block.MAXNZ;
    		temp_blocks[b].JA = d_JA;
    		temp_blocks[b].AS = d_AS;
    	}
    	// Copia l'intero array di blocchi sulla GPU
    	CUDA_CHECK(cudaMemcpy(d_blocks, temp_blocks, hll_mat.num_blocks * sizeof(ELLPACKBlock), cudaMemcpyHostToDevice));
       // Trova il valore massimo di MAXNZ tra tutti i blocchi
		int max_MAXNZ = compute_max_MAXNZ(hll_mat.blocks, hll_mat.num_blocks);
		// Funzione lambda per calcolare la shared memory in base alla dimensione del blocco
		auto sharedMemCalculator_hll = [&](int b_size) {
    		int warps = b_size / 32;
    		return warps * max_MAXNZ * sizeof(double);
		};
    	double avg_time_seconds_warp_shared_hll;
    	double avg_flops_warp_shared_hll;
		// Trova la dimensione ottimale del blocco considerando la shared memory
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGrid, &blockSize,
                                             spmv_hll_warp_shared_kernel_v1,
                                             sharedMemCalculator_hll, 0);

		if(blockSize != 0){
			// Assicurati che blockSize sia multiplo di 32
			blockSize = (blockSize / 32) * 32;
			warps_per_block = blockSize / 32;

			// Calcola il numero di blocchi necessari
			int grid_size = (csr_mat.M + warps_per_block - 1) / warps_per_block;

			// Calcola la memoria shared effettivamente necessaria
			size_t shared_mem_size = warps_per_block * max_MAXNZ * sizeof(double);

			// Verifica che la dimensione della shared memory non superi il limite
			int max_shared_mem_int;
			cudaDeviceGetAttribute(&max_shared_mem_int, cudaDevAttrMaxSharedMemoryPerBlock, 0);
			size_t max_shared_mem = static_cast<size_t>(max_shared_mem_int);
			if (shared_mem_size > max_shared_mem) {
    		printf("Errore: La shared memory richiesta (%zu bytes) supera il limite (%zu bytes)\n",
          		shared_mem_size, max_shared_mem);
			}

			// ==================================================
	        // TEST 1: VERSIONE WARP PER RIGA E MEMORIA CONDIVISA (HLL)
	        // ==================================================
    		printf("\n--- TEST WARP PER RIGA CON MEMORIA CONDIVISA E HLL ---\n");

    		clear_gpu_cache(64);
	        for(int i = 0; i < NUM_ITERATION; i++) {
	            printf("Iterazione numero: %d\n", i);
	            CUDA_CHECK(cudaMemset(d_y_hll, 0, csr_mat.M * sizeof(double)));
	            // Registra l'evento di inizio
	            CUDA_CHECK(cudaEventRecord(start));

        		spmv_hll_warp_shared_kernel_v1<<<grid_size, blockSize, shared_mem_size>>>(
	            csr_mat.M, d_blocks, d_x_hll, d_y_hll);

	            CUDA_CHECK(cudaGetLastError());

	            // Registra l'evento di fine
	            CUDA_CHECK(cudaEventRecord(stop));

	            // Aspetta la fine dell'esecuzione
	            CUDA_CHECK(cudaEventSynchronize(stop));

	            // Calcola il tempo trascorso
	            float milliseconds = 0;
	            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        		update_medium_metric(WARP_SHARED_MEMORY_HLL_TIME, milliseconds / 1000.0f);

	            printf("Tempo impiegato per l'esecuzione del Kernel-Warp con shared memory: %f secondi\n", milliseconds/1000.0f);
	            double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
	            printf("Tempo esecuzione parallelo Warp con shared memory CSR: %f secondi\n", milliseconds/1000.0f);
	            printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
	            print_flops(flops_parallel);

	            // Copia il risultato ad ogni iterazione
	            CUDA_CHECK(cudaMemcpy(y_hll, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

	            // Controlla la correttezza del risultato ad ogni iterazione
	            printf("Verifica risultato iterazione %d:\n", i);
	            bool error_warp = checkDifferences(y_serial, y_hll, csr_mat.M, false);

	        }
	        // Calcola e stampa il tempo medio in secondi
	        avg_time_seconds_warp_shared_hll = get_metric_value(WARP_SHARED_MEMORY_HLL_TIME) / NUM_ITERATION;
	        printf("Tempo medio impiegato per l'esecuzione del Kernel-Warp con Shared Memory HLL: %f secondi\n", avg_time_seconds_warp_shared_hll);
			avg_flops_warp_shared_hll = calculate_flops(csr_mat.nz, avg_time_seconds_warp_shared_hll);
	        printf("Flops medi: ");
	        print_flops(avg_flops_warp_shared_hll);
		}

    	// ================================
    	// TEST 1: VERSIONE SERIALE (HLL)
    	// ================================
    	printf("\n--- TEST SERIALE HLL ---\n");
    	clear_gpu_cache(64);
    	for(int i = 0; i < NUM_ITERATION; i++){
    		printf("Iterazione numero: %d\n", i+1);
    		memset(y_hll, 0, csr_mat.M * sizeof(double));

    		cudaEventRecord(start);
    		spmv_hll_serial(hll_mat.num_blocks, hll_mat.blocks, x, y_hll);
    		cudaEventRecord(stop);

    		// Aspetta la fine dell'esecuzione
    		cudaEventSynchronize(stop);

    		// Calcola il tempo trascorso
    		float milliseconds_hll = 0;
    		cudaEventElapsedTime(&milliseconds_hll, start, stop);
    		update_medium_metric(SERIAL_HLL_TIME, milliseconds_hll / 1000.0f);

    		double flops_serial_hll = calculate_flops(csr_mat.nz, milliseconds_hll/1000.0f);
    		printf("Tempo impiegato per l'esecuzione seriale HLL: %f secondi\n", milliseconds_hll/1000.0f);
    		printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_serial_hll);
    		print_flops(flops_serial_hll);

            printf("Verifica risultato iterazione %d:\n", i);
            bool error_warp = checkDifferences(y_serial, y_hll, csr_mat.M, false);
    	}

    	double avg_time_seconds_serial_hll = get_metric_value(SERIAL_HLL_TIME) / NUM_ITERATION;
    	printf("Tempo medio impiegato per l'esecuzione seriale HLL: %f secondi\n", avg_time_seconds_serial_hll);
		double avg_flops_hll_serial = calculate_flops(csr_mat.nz, avg_time_seconds_serial_hll);
        printf("Flops medi: ");
        print_flops(avg_flops_hll_serial);

		cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                  spmv_hll_naive_kernel, 0, 0);

		// Non è necessario forzare multipli di 32 come nella versione warp
		// poiché ogni thread elabora una riga indipendentemente
		int threadsPerBlock_hll = blockSize;
		int blocksPerGrid_hll = (csr_mat.M + threadsPerBlock_hll - 1) / threadsPerBlock_hll;

		// ======================================
		// TEST 2: VERSIONE HLL NAIVE
		// ======================================
    	printf("\n--- TEST RIGA PER THREAD HLL ---\n");
    	clear_gpu_cache(64);
		for(int i = 0; i < NUM_ITERATION; i++) {
		    printf("Iterazione numero: %d\n", i+1);

		    // Azzera il vettore risultato
		    CUDA_CHECK(cudaMemset(d_y_hll, 0, csr_mat.M * sizeof(double)));

		    // Registra l'evento di inizio
		    CUDA_CHECK(cudaEventRecord(start));

		    // Lancio del kernel CUDA con la struttura HLL
		    spmv_hll_naive_kernel<<<blocksPerGrid_hll, threadsPerBlock_hll>>>(
		        csr_mat.M, d_blocks, d_x_hll, d_y_hll
		    );

            CUDA_CHECK(cudaGetLastError());

            // Registra l'evento di fine
            CUDA_CHECK(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            CUDA_CHECK(cudaEventSynchronize(stop));

		    // Calcola il tempo trascorso
		    float milliseconds = 0;
		    cudaEventElapsedTime(&milliseconds, start, stop);
			update_medium_metric(ROW_HLL_TIME, milliseconds / 1000.0f);

		    printf("Tempo impiegato per l'esecuzione del Kernel-HLL-Naive: %f secondi\n", milliseconds/1000.0f);

		    // Usa il conteggio nz della matrice CSR per i confronti
		    double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);

		    printf("Tempo esecuzione parallelo HLL: %f secondi\n", milliseconds/1000.0f);
		    printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
		    print_flops(flops_parallel);

		    // Copia il risultato ad ogni iterazione
			CUDA_CHECK(cudaMemcpy(y_hll, d_y_hll, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

		    // Controlla la correttezza del risultato ad ogni iterazione
		    printf("Verifica risultato iterazione %d:\n", i+1);
		    bool error_hll = checkDifferences(y_serial, y_hll, csr_mat.M, false);
		}

		// Calcola e stampa il tempo medio in secondi
		double avg_time_seconds_hll_naive = get_metric_value(ROW_HLL_TIME) / NUM_ITERATION;
		printf("Tempo medio impiegato per l'esecuzione del Kernel-HLL-Naive: %f secondi\n", avg_time_seconds_hll_naive);
		double avg_flops_hll = calculate_flops(csr_mat.nz, avg_time_seconds_hll_naive);
        printf("Flops medi: ");
        print_flops(avg_flops_hll);

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                  spmv_hll_warp_kernel, 0, 0);

	// Assicurati che blockSize sia multiplo di 32 (dimensione di un warp)
	blockSize = (blockSize / 32) * 32;
	warpsPerBlock = blockSize / 32;
	int threadsPerBlock_hll_warp = blockSize;

	// Un warp per riga, quindi M warps totali
	int blocksPerGrid_hll_warp = (csr_mat.M + warpsPerBlock - 1) / warpsPerBlock;

    // =========================
    // TEST 2: VERSIONE HLL WARP
    // =========================
    printf("\n--- TEST WARP PER RIGA HLL ---\n");
    clear_gpu_cache(64);
    for(int i = 0; i < NUM_ITERATION; i++) {
        printf("Iterazione HLL-Warp numero: %d\n", i+1);
        CUDA_CHECK(cudaMemset(d_y_hll, 0, csr_mat.M * sizeof(double)));
        CUDA_CHECK(cudaEventRecord(start));

        spmv_hll_warp_kernel<<<blocksPerGrid_hll_warp, threadsPerBlock_hll_warp>>>(
            csr_mat.M,        // total_M
            d_blocks,         // Puntatore ai blocchi su GPU
            d_x_hll,          // Vettore x su GPU
            d_y_hll          // Vettore y su GPU
        );

       	CUDA_CHECK(cudaGetLastError());

       	// Registra l'evento di fine
       	CUDA_CHECK(cudaEventRecord(stop));

       	// Aspetta la fine dell'esecuzione
       	CUDA_CHECK(cudaEventSynchronize(stop));


       	float milliseconds = 0;
       	CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
       	printf("Tempo Kernel-HLL-Warp: %f ms\n", milliseconds/1000.0f);
    	update_medium_metric(WARP_HLL_TIME, milliseconds / 1000.0f);

    	double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);

    	printf("Tempo esecuzione parallelo HLL: %f secondi\n", milliseconds/1000.0f);
    	printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
    	print_flops(flops_parallel);

        CUDA_CHECK(cudaMemcpy(y_hll, d_y_hll, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));
        printf("Verifica risultato HLL-Warp iterazione %d:\n", i+1);
        checkDifferences(y_serial, y_hll, csr_mat.M, true); // Abilita stampa errori
    }

    double avg_time_hll_warp = get_metric_value(WARP_HLL_TIME) / NUM_ITERATION;
    printf("Tempo medio Kernel-HLL-Warp: %f secondi\n", avg_time_hll_warp);
    double avg_flops_hll_warp = calculate_flops(csr_mat.nz, avg_time_hll_warp);
    printf("Flops medi: ");
    print_flops(avg_flops_hll_warp);

    for (int b = 0; b < hll_mat.num_blocks; b++) {
    	CUDA_CHECK(cudaFree(temp_blocks[b].JA));
    	CUDA_CHECK(cudaFree(temp_blocks[b].AS));
    }

    // Output risultati medi e salvataggio CSV
    printf("\n========== Risultati medi per matrice %s ==========\n",
		dir->d_name);

    printf("--- Tempi di esecuzione ---\n");
    printf("Tempo medio seriale CSR: %f secondi\n", avg_time_seconds_serial);
    printf("Tempo medio seriale HLL: %f secondi\n", avg_time_seconds_serial_hll);
    printf("Tempo medio CSR thread per riga: %f secondi\n", avg_time_seconds_naive);
    printf("Tempo medio CSR parallelo warp per riga: %f secondi\n", avg_time_seconds_warp);
    printf("Tempo medio CSR parallelo warp per riga con shared memory: %f secondi\n", avg_time_seconds_warp_shared);
   	printf("Tempo medio HLL thread per riga: %f secondi\n", avg_time_seconds_hll_naive);
    printf("Tempo medio parallelo warp per riga HLL: %f secondi\n", avg_time_hll_warp);
    printf("Tempo medio parallelo warp per riga con Shared Memory HLL: %f secondi\n", avg_time_seconds_warp_shared_hll);

    printf("\n--- Prestazioni (FLOPS) ---\n");
    printf("Prestazioni seriali CSR: ");
    print_flops(avg_flops_serial);
    printf("Prestazioni seriali HLL: ");
    print_flops(avg_flops_hll_serial);
    printf("Prestazioni parallele CSR THREAD PER RIGA: ");
    print_flops(avg_flops_parallel);
    printf("Prestazioni parallele CSR WARP PER RIGA: ");
    print_flops(avg_flops_warp);
    printf("Prestazioni parallele CSR WARP PER RIGA CON SHARED MEMORY: ");
    print_flops(avg_flops_warp_shared);
    printf("Prestazioni parallele HLL THREAD PER RIGA: ");
    print_flops(avg_flops_hll);
    printf("Prestazioni parallele HLL WARP PER RIGA: ");
   	print_flops(avg_flops_hll_warp);
    printf("Prestazioni parallele CSR WARP PER RIGA CON SHARED MEM HLL: ");
    print_flops(avg_flops_warp_shared_hll);


    write_results_to_csv(dir->d_name, csr_mat.M, csr_mat.N, csr_mat.nz,
						avg_time_seconds_serial, avg_time_seconds_serial_hll, avg_time_seconds_naive,
						avg_time_seconds_warp,avg_time_seconds_warp_shared, avg_time_seconds_warp_shared_hll, avg_time_seconds_hll_naive, avg_time_hll_warp,
						avg_flops_serial, avg_flops_hll_serial, avg_flops_parallel,
						avg_flops_warp, avg_flops_hll, avg_flops_hll_warp, avg_flops_warp_shared, avg_flops_warp_shared_hll,
						output_csv);

    FREE_CHECK(temp_blocks);
    CUDA_CHECK(cudaFree(d_blocks));
    CUDA_CHECK(cudaFree(d_x_hll));
    CUDA_CHECK(cudaFree(d_y_hll));
    FREE_CHECK(y_serial);
    FREE_CHECK(y_parallel_csr);
    FREE_CHECK(y_hll);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

	}
}

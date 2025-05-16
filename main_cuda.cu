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
#include <helper_cuda.h>

#define NUM_ITERATION 95 + ITERATION_SKIP


int main() {

    struct dirent *dir;
    const char *directory_path = "../matrix_for_test/";
    //const char *directory_path = "../matrix_generated/";
	const char *output_csv = "../result/spmv_results_cuda.csv";
    const char *directory_result = "../result";

    create_directory(directory_result);

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

        // Inizializzazione vettore y
        double *y_serial = (double *)malloc(csr_mat.M * sizeof(double));
        double *y_parallel_csr = (double *)malloc(csr_mat.M * sizeof(double));
        double *y_hll = (double *)malloc(csr_mat.M * sizeof(double));

        // ================================
        // TEST 1: VERSIONE SERIALE (CSR)
        // ================================
        cudaEvent_t start, stop;
    	checkCudaErrors(cudaEventCreate(&start));
    	checkCudaErrors(cudaEventCreate(&stop));

        // Esecuzione seriale per confronto
    	printf("\n--- TEST SERIALE CSR ---\n");
    	reset_medium_time_metrics();
        for(int i = 1; i < NUM_ITERATION; i++){
            printf("Iterazione numero: %d\n", i);
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
			if(i > ITERATION_SKIP){
            //aggiorna la metrica
        	update_medium_metric(SERIAL_TIME, milliseconds / 1000.0f);

            double flops_serial = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
			printf("Tempo impiegato per l'esecuzione seriale CSR: %f secondi\n", milliseconds/1000.0f);
			printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_serial);
            print_flops(flops_serial);
            }
		}

        PerformanceMetrics perf_metrics_serial_csr;
        perf_metrics_serial_csr.time = get_metric_value(SERIAL_TIME);
        perf_metrics_serial_csr.flops = calculate_flops(csr_mat.nz, perf_metrics_serial_csr.time);
    	printf("Tempo medio impiegato per l'esecuzione seriale CSR: %f secondi\n", perf_metrics_serial_csr.time);
    	printf("Valore medio dei FLOPS: ");
    	print_flops(perf_metrics_serial_csr.flops);
    	// Dichiarazione dei puntatori per i dati sulla GPU
    	int *d_row_ptr;
    	int *d_col_idx;
    	double *d_values;
    	double *d_x;
    	double *d_y;

    	// Allocazione memoria sulla GPU
    	checkCudaErrors(cudaMalloc((void **)&d_row_ptr, (csr_mat.M + 1) * sizeof(int)));
    	checkCudaErrors(cudaMalloc((void **)&d_col_idx, csr_mat.nz * sizeof(int)));
    	checkCudaErrors(cudaMalloc((void **)&d_values, csr_mat.nz * sizeof(double)));
    	checkCudaErrors(cudaMalloc((void **)&d_x, csr_mat.N * sizeof(double)));
    	checkCudaErrors(cudaMalloc((void **)&d_y, csr_mat.M * sizeof(double)));

    	// Copia dei dati dalla CPU alla GPU
    	checkCudaErrors(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (csr_mat.M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_col_idx, csr_mat.col_idx, csr_mat.nz * sizeof(int), cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_values, csr_mat.values, csr_mat.nz * sizeof(double), cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_x, x, csr_mat.N * sizeof(double), cudaMemcpyHostToDevice));

        // Configurazione dei parametri per il lancio del kernel
        int minGrid, blockSize, blocksPerGrid, threadsPerBlock;
		cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, spmv_csr_naive_kernel, 0, csr_mat.M);
		blocksPerGrid   = (csr_mat.M + blockSize - 1) / blockSize;
		threadsPerBlock = blockSize;

        // ======================================
        // TEST 2: VERSIONE THREAD PER RIGA (CSR)
        // ======================================
    	printf("\n--- TEST THREAD PER RIGA CSR ---\n");
        reset_medium_time_metrics();
        clear_gpu_cache(64);
    	for(int i = 1; i < NUM_ITERATION; i++) {
            printf("Iterazione numero: %d\n", i);
            checkCudaErrors(cudaMemset(d_y, 0, csr_mat.M * sizeof(double)));
            // Registra l'evento di inizio
            checkCudaErrors(cudaEventRecord(start));

            // Lancio del kernel CUDA
            spmv_csr_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                csr_mat.M, d_row_ptr, d_col_idx, d_values, d_x, d_y
            );

            checkCudaErrors(cudaGetLastError());

            // Registra l'evento di fine
            checkCudaErrors(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            checkCudaErrors(cudaEventSynchronize(stop));

            // Calcola il tempo trascorso
            float milliseconds = 0;
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    		// Copia il risultato ad ogni iterazione
    		checkCudaErrors(cudaMemcpy(y_parallel_csr, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

    		// Controlla la correttezza del risultato ad ogni iterazione
    		printf("Verifica risultato iterazione %d:\n", i);
    		DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_parallel_csr, csr_mat.M);
			accumulateErrors(&diffMetrics, ROW_CSR_TIME);

            if(i > ITERATION_SKIP){
            	update_medium_metric(ROW_CSR_TIME, milliseconds / 1000.0f);

            	printf("Tempo impiegato per l'esecuzione del Kernel-Naive: %f secondi\n", milliseconds/1000.0f);
            	double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
            	printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
            	print_flops(flops_parallel);
                printf("Errore relativo: %f\n", diffMetrics.mean_rel_err);
                printf("Errore assoluto: %f\n", diffMetrics.mean_abs_err);
            }
	    }
        PerformanceMetrics perf_metrics_naive_csr;
    	// Calcola e stampa il tempo medio in secondi
    	perf_metrics_naive_csr.time = get_metric_value(ROW_CSR_TIME);
        perf_metrics_naive_csr.flops = calculate_flops(csr_mat.nz, perf_metrics_naive_csr.time);
    	printf("Tempo medio impiegato per l'esecuzione del Kernel-Naive CSR: %f secondi\n", perf_metrics_naive_csr.time);
		printf("FLOPS medi: ");
        print_flops(perf_metrics_naive_csr.flops);
		DiffMetrics mediumCsrParallel = computeAverageErrors(ROW_CSR_TIME);
        printf("Errore relativo medio parallelo CSR: %f\n", mediumCsrParallel.mean_rel_err);
        printf("Errore assoluto medio parallelo CSR: %f\n", mediumCsrParallel.mean_abs_err);

        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                  spmv_csr_warp_kernel, 0, 0);

		// Assicurati che blockSize sia multiplo di 32 (dimensione di un warp)
		blockSize = (blockSize / 32) * 32;
        if (blockSize < 32) blockSize = 32;
		int warps_per_block = blockSize / 32;
		int threadsPerBlock_warp = blockSize;

		// Un warp per riga, quindi M warps totali
		int blocksPerGrid_warp = (csr_mat.M + warps_per_block - 1) / warps_per_block; //Devo distribuire le righe
                //su un certo numero di blocchi, effettuo quindi divisione intero

        // ======================================
        // TEST 3: VERSIONE WARP PER RIGA (CSR)
        // ======================================
    	printf("\n--- TEST WARP PER RIGA CSR ---\n");
        reset_medium_time_metrics();
    	clear_gpu_cache(64);
        for(int i = 1; i < NUM_ITERATION; i++) {
            printf("Iterazione numero: %d\n", i);
            checkCudaErrors(cudaMemset(d_y, 0, csr_mat.M * sizeof(double)));
            // Registra l'evento di inizio
            checkCudaErrors(cudaEventRecord(start));

            // Lancio del kernel CUDA warp-based
            spmv_csr_warp_kernel<<<blocksPerGrid_warp, threadsPerBlock_warp>>>(
                csr_mat.M, d_row_ptr, d_col_idx, d_values, d_x, d_y
            );

            checkCudaErrors(cudaGetLastError());

            // Registra l'evento di fine
            checkCudaErrors(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            checkCudaErrors(cudaEventSynchronize(stop));

            // Calcola il tempo trascorso
            float milliseconds = 0;
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

        	// Copia il risultato ad ogni iterazione
        	checkCudaErrors(cudaMemcpy(y_parallel_csr, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

        	// Controlla la correttezza del risultato ad ogni iterazione
        	printf("Verifica risultato iterazione %d:\n", i);
        	DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_parallel_csr, csr_mat.M);
			accumulateErrors(&diffMetrics, WARP_CSR_TIME);

            if(i > ITERATION_SKIP){
            	update_medium_metric(WARP_CSR_TIME, milliseconds / 1000.0f);
            	printf("Tempo impiegato per l'esecuzione del Kernel-Warp: %f secondi\n", milliseconds/1000.0f);
            	double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
            	printf("Tempo esecuzione parallelo Warp CSR: %f secondi\n", milliseconds/1000.0f);
            	printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
            	print_flops(flops_parallel);
			}
        }
        PerformanceMetrics perf_metrics_warp_csr;
        // Calcola e stampa il tempo medio in secondi
        perf_metrics_warp_csr.time = get_metric_value(WARP_CSR_TIME);
        printf("Tempo medio impiegato per l'esecuzione del Kernel-Warp CSR: %f secondi\n", perf_metrics_warp_csr.time);
		perf_metrics_warp_csr.flops = calculate_flops(csr_mat.nz, perf_metrics_warp_csr.time);
        printf("Flops medi: ");
        print_flops(perf_metrics_warp_csr.flops);
        DiffMetrics mediumCsrWarp = computeAverageErrors(WARP_CSR_TIME);
        printf("Errore relativo medio parallelo CSR: %f\n", mediumCsrWarp.mean_rel_err);
        printf("Errore assoluto medio parallelo CSR: %f\n", mediumCsrWarp.mean_abs_err);

        // ==================================================
        // TEST 4: VERSIONE WARP PER RIGA E MEMORIA CONDIVISA (CSR)
        // ==================================================
        int cacheElements = min(csr_mat.N, MAX_CACHE);
		size_t sharedMemSize = sizeof(double) * cacheElements;

		// Funzione lambda per calcolare la shared memory in base alla dimensione del blocco
		auto sharedMemCalculator = [&](int blockSize) {
    		return sharedMemSize;
		};

		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGrid, &blockSize,
                                             spmv_csr_warp_shared_memory_kernel,
                                             sharedMemCalculator, 0);
        //Dobbiamo gestire bene il numero di blocchi attivi e il tradeoff con la mem. condivisa, abbiamo infatti un
                //limite della mem. condivisa disponibile, quindi la quantità assegnata ad ogni blocco limita
                //necessariamente il numero di blocchi attivi

		blockSize = (blockSize +31 / 32) * 32;
        if (blockSize < 32) blockSize = 32;
		int warpsPerBlock = blockSize / 32;
		threadsPerBlock = blockSize;

		// Un warp per riga, quindi M warps totali
		blocksPerGrid = (csr_mat.M + warpsPerBlock - 1) / warpsPerBlock;
        printf("\n--- TEST WARP PER RIGA CON MEMORIA CONDIVISA E CSR ---\n");
		reset_medium_time_metrics();
        clear_gpu_cache(64);
        for(int i = 1; i < NUM_ITERATION; i++) {
            printf("Iterazione numero: %d\n", i);
            checkCudaErrors(cudaMemset(d_y, 0, csr_mat.M * sizeof(double)));
            // Registra l'evento di inizio
            checkCudaErrors(cudaEventRecord(start));

            // Lancio del kernel CUDA warp-based con shared memory
            spmv_csr_warp_shared_memory_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                csr_mat.M, d_row_ptr, d_col_idx, d_values, d_x, d_y, cacheElements
            );

            checkCudaErrors(cudaGetLastError());

            // Registra l'evento di fine
            checkCudaErrors(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            checkCudaErrors(cudaEventSynchronize(stop));

            // Calcola il tempo trascorso
            float milliseconds = 0;
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

        	// Copia il risultato ad ogni iterazione
            memset(y_parallel_csr, 0, csr_mat.M * sizeof(double));

        	checkCudaErrors(cudaMemcpy(y_parallel_csr, d_y, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

        	// Controlla la correttezza del risultato ad ogni iterazione
        	printf("Verifica risultato iterazione %d:\n", i);
        	DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_parallel_csr, csr_mat.M);

			accumulateErrors(&diffMetrics, WARP_SHARED_MEMORY_CSR_TIME);

            if (i > ITERATION_SKIP){
        		update_medium_metric(WARP_SHARED_MEMORY_CSR_TIME, milliseconds / 1000.0f);
            	printf("Tempo impiegato per l'esecuzione del Kernel-Warp com shared memory CSR: %f secondi\n", milliseconds/1000.0f);
            	double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
            	printf("Tempo esecuzione parallelo Warp shared memory CSR: %f secondi\n", milliseconds/1000.0f);
            	printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
            	print_flops(flops_parallel);
                printf("Errore relativo: %f\n", diffMetrics.mean_rel_err);
				printf("Errore assoluto: %f\n", diffMetrics.mean_abs_err);
        	}
        }

        PerformanceMetrics perf_metrics_warp_shared_csr;
        // Calcola e stampa il tempo medio in secondi
        perf_metrics_warp_shared_csr.time = get_metric_value(WARP_SHARED_MEMORY_CSR_TIME);
        printf("Tempo medio impiegato per l'esecuzione del Kernel-Warp shared memory CSR: %f secondi\n", perf_metrics_warp_shared_csr.time);
		perf_metrics_warp_shared_csr.flops = calculate_flops(csr_mat.nz, perf_metrics_warp_shared_csr.time);
        printf("Flops medi: ");
        print_flops(perf_metrics_warp_shared_csr.flops);
        DiffMetrics mediumCsrWarpShared = computeAverageErrors(WARP_SHARED_MEMORY_CSR_TIME);
        printf("Errore relativo medio Kernel-Warp shared memory CSR: %f\n", mediumCsrWarpShared.mean_rel_err);
        printf("Errore assoluto medio Kernel-Warp shared memory CSR: %f\n", mediumCsrWarpShared.mean_abs_err);


    	// Vettori x e y sulla device
    	double *d_x_hll;
    	double *d_y_hll;
    	checkCudaErrors(cudaMalloc((void **)&d_x_hll, csr_mat.N * sizeof(double)));
    	checkCudaErrors(cudaMalloc((void **)&d_y_hll, csr_mat.M * sizeof(double)));
    	// Copia dei dati x dalla CPU alla GPU
    	checkCudaErrors(cudaMemcpy(d_x_hll, x, csr_mat.N * sizeof(double), cudaMemcpyHostToDevice));
    	// Alloca array di blocchi ELLPACK sulla GPU
    	ELLPACKBlock *d_blocks;
    	checkCudaErrors(cudaMalloc((void **)&d_blocks, hll_mat.num_blocks * sizeof(ELLPACKBlock)));
    	// Crea array temporaneo di blocchi sull'host
    	ELLPACKBlock *temp_blocks = (ELLPACKBlock *)malloc(hll_mat.num_blocks * sizeof(ELLPACKBlock));
    	// Alloca e copia ogni blocco ELLPACK
    	for (int b = 0; b < hll_mat.num_blocks; b++) {
    		ELLPACKBlock h_block = hll_mat.blocks[b];

    		// Allocazione memoria per i dati del blocco (layout trasposto)
    		int *d_JA;
    		double *d_AS;
    		checkCudaErrors(cudaMalloc((void **)&d_JA, h_block.MAXNZ * h_block.M * sizeof(int)));
    		checkCudaErrors(cudaMalloc((void **)&d_AS, h_block.MAXNZ * h_block.M * sizeof(double)));

    		// Copia dei dati del blocco
    		checkCudaErrors(cudaMemcpy(d_JA, h_block.JA, h_block.MAXNZ * h_block.M * sizeof(int), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(d_AS, h_block.AS, h_block.MAXNZ * h_block.M * sizeof(double), cudaMemcpyHostToDevice));

    		// Imposta i campi del blocco temporaneo
    		temp_blocks[b].M = h_block.M;
    		temp_blocks[b].N = h_block.N;
    		temp_blocks[b].MAXNZ = h_block.MAXNZ;
    		temp_blocks[b].JA = d_JA;
    		temp_blocks[b].AS = d_AS;
    	}
    	// Copia l'intero array di blocchi sulla GPU
    	checkCudaErrors(cudaMemcpy(d_blocks, temp_blocks, hll_mat.num_blocks * sizeof(ELLPACKBlock), cudaMemcpyHostToDevice));
        // Trova il valore massimo di MAXNZ tra tutti i blocchi
		int max_MAXNZ = compute_max_MAXNZ(hll_mat.blocks, hll_mat.num_blocks);
		// Funzione lambda per calcolare la shared memory in base alla dimensione del blocco
		auto sharedMemCalculator_hll = [&](int b_size) {
    		int warps = b_size / 32;
    		return warps * max_MAXNZ * sizeof(double);
		};

		// Trova la dimensione ottimale del blocco considerando la shared memory
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGrid, &blockSize,
                                             spmv_hll_warp_shared_kernel_v1,
                                             sharedMemCalculator_hll, 0);

    	PerformanceMetrics perf_metrics_warp_shared_hll;
        DiffMetrics mediumHllWarpShared;
		if(blockSize != 0){
			// Assicurati che blockSize sia multiplo di 32
			blockSize = ((blockSize + 31) / 32) * 32;
			if (blockSize < 32) blockSize = 32;
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
			reset_medium_time_metrics();
        	clear_gpu_cache(64);
	        for(int i = 1; i < NUM_ITERATION; i++) {
	            printf("Iterazione numero: %d\n", i);
	            checkCudaErrors(cudaMemset(d_y_hll, 0, csr_mat.M * sizeof(double)));
                double* h_y = (double*)malloc(csr_mat.M * sizeof(double));
	            // Registra l'evento di inizio
	            checkCudaErrors(cudaEventRecord(start));

        		spmv_hll_warp_shared_kernel_v1<<<grid_size, blockSize, shared_mem_size>>>(
	            csr_mat.M, d_blocks, d_x_hll, d_y_hll);

	            checkCudaErrors(cudaGetLastError());

	            // Registra l'evento di fine
	            checkCudaErrors(cudaEventRecord(stop));

	            // Aspetta la fine dell'esecuzione
	            checkCudaErrors(cudaEventSynchronize(stop));

	            // Calcola il tempo trascorso
	            float milliseconds = 0;
	            checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
                // Copia il risultato ad ogni iterazione
                memset(y_hll, 0, csr_mat.M * sizeof(double));
                for(int j = 0; j < 30; j++) {
                    printf("%.10f ", y_hll[j]);
                }
	            checkCudaErrors(cudaMemcpy(y_hll, d_y_hll, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));
                for(int j = 0; j < 30; j++) {
                    printf("%.10f ", y_hll[j]);
                }
	            // Controlla la correttezza del risultato ad ogni iterazione
	            printf("Verifica risultato iterazione %d:\n", i);
	        	DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M);

				accumulateErrors(&diffMetrics, WARP_SHARED_MEMORY_HLL_TIME);
        		if(i > ITERATION_SKIP){
                	update_medium_metric(WARP_SHARED_MEMORY_HLL_TIME, milliseconds / 1000.0f);
	            	printf("Tempo impiegato per l'esecuzione del Kernel-Warp con shared memory HLL: %f secondi\n", milliseconds/1000.0f);
	            	double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
	            	printf("Tempo esecuzione parallelo Warp con shared memory HLL: %f secondi\n", milliseconds/1000.0f);
	            	printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
	            	print_flops(flops_parallel);
                    printf("Errore relativo: %f\n", diffMetrics.mean_rel_err);
                    printf("Errore assoluto: %f\n", diffMetrics.mean_abs_err);
				}
	        }

	        // Calcola e stampa il tempo medio in secondi
	        perf_metrics_warp_shared_hll.time = get_metric_value(WARP_SHARED_MEMORY_HLL_TIME);
	        printf("Tempo medio impiegato per l'esecuzione del Kernel-Warp con Shared Memory HLL: %f secondi\n", perf_metrics_warp_shared_hll.time);
			perf_metrics_warp_shared_hll.flops = calculate_flops(csr_mat.nz, perf_metrics_warp_shared_hll.time);
	        printf("Flops medi: ");
	        print_flops(perf_metrics_warp_shared_hll.flops);
            mediumHllWarpShared = computeAverageErrors(WARP_SHARED_MEMORY_HLL_TIME);
            printf("Errore relativo medio parallelo HLL Warp con Shared Memory: %f\n", mediumHllWarpShared.mean_rel_err);
            printf("Errore assoluto medio parallelo HLL Warp con Shared Memory: %f\n", mediumHllWarpShared.mean_abs_err);

		}

    	// ================================
    	// TEST 1: VERSIONE SERIALE (HLL)
    	// ================================
    	printf("\n--- TEST SERIALE HLL ---\n");
    	reset_medium_time_metrics();
        clear_gpu_cache(64);
    	for(int i = 1; i < NUM_ITERATION; i++){
    		printf("Iterazione numero: %d\n", i);
    		memset(y_hll, 0, csr_mat.M * sizeof(double));

    		checkCudaErrors(cudaEventRecord(start));
    		spmv_hll_serial(hll_mat.num_blocks, hll_mat.blocks, x, y_hll);
    		checkCudaErrors(cudaEventRecord(stop));
    		// Aspetta la fine dell'esecuzione
    		checkCudaErrors(cudaEventSynchronize(stop));
            float milliseconds_hll = 0;
    		checkCudaErrors(cudaEventElapsedTime(&milliseconds_hll, start, stop));
    		printf("Verifica risultato iterazione %d:\n", i);
    		DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M);
			accumulateErrors(&diffMetrics, SERIAL_HLL_TIME);
            if(i > ITERATION_SKIP){
              	// Calcola il tempo trascorso
    			update_medium_metric(SERIAL_HLL_TIME, milliseconds_hll / 1000.0f);

    			double flops_serial_hll = calculate_flops(csr_mat.nz, milliseconds_hll/1000.0f);
    			printf("Tempo impiegato per l'esecuzione seriale HLL: %f secondi\n", milliseconds_hll/1000.0f);
    			printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_serial_hll);
    			print_flops(flops_serial_hll);
                printf("Errore relativo: %f\n", diffMetrics.mean_rel_err);
                printf("Errore assoluto: %f\n", diffMetrics.mean_abs_err);
    		}
    	}
		PerformanceMetrics perf_metrics_serial_hll;
    	perf_metrics_serial_hll.time = get_metric_value(SERIAL_HLL_TIME);
    	printf("Tempo medio impiegato per l'esecuzione seriale HLL: %f secondi\n", perf_metrics_serial_hll.time);
		perf_metrics_serial_hll.flops = calculate_flops(csr_mat.nz, perf_metrics_serial_hll.time);
        printf("Flops medi: ");
        print_flops(perf_metrics_serial_hll.flops);
        DiffMetrics mediumHllSerial = computeAverageErrors(SERIAL_HLL_TIME);
        printf("Errore relativo medio HLL seriale: %f\n", mediumHllSerial.mean_rel_err);
        printf("Errore assoluto medio Hll seriale: %f\n", mediumHllSerial.mean_abs_err);


		cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                  spmv_hll_naive_kernel, 0, 0);


		int threadsPerBlock_hll = blockSize;
		int blocksPerGrid_hll = (csr_mat.M + threadsPerBlock_hll - 1) / threadsPerBlock_hll;

		// ======================================
		// TEST 2: VERSIONE HLL NAIVE
		// ======================================
    	printf("\n--- TEST RIGA PER THREAD HLL ---\n");
    	reset_medium_time_metrics();
        clear_gpu_cache(64);
		for(int i = 1; i < NUM_ITERATION; i++) {
		    printf("Iterazione numero: %d\n", i);

		    // Azzera il vettore risultato
		    checkCudaErrors(cudaMemset(d_y_hll, 0, csr_mat.M * sizeof(double)));

		    // Registra l'evento di inizio
		    checkCudaErrors(cudaEventRecord(start));

		    // Lancio del kernel CUDA con la struttura HLL
		    spmv_hll_naive_kernel<<<blocksPerGrid_hll, threadsPerBlock_hll>>>(
		        csr_mat.M, d_blocks, d_x_hll, d_y_hll
		    );

            checkCudaErrors(cudaGetLastError());

            // Registra l'evento di fine
            checkCudaErrors(cudaEventRecord(stop));

            // Aspetta la fine dell'esecuzione
            checkCudaErrors(cudaEventSynchronize(stop));

			// Controlla la correttezza del risultato ad ogni iterazione
			printf("Verifica risultato iterazione %d:\n", i);
            float milliseconds = 0;
		    cudaEventElapsedTime(&milliseconds, start, stop);
            // Copia il risultato ad ogni iterazione
			checkCudaErrors(cudaMemcpy(y_hll, d_y_hll, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

			DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M);
			accumulateErrors(&diffMetrics, ROW_HLL_TIME);
		    // Calcola il tempo trascorso
			if(i > ITERATION_SKIP){
            update_medium_metric(ROW_HLL_TIME, milliseconds / 1000.0f);
		    printf("Tempo impiegato per l'esecuzione del Kernel-HLL-Naive: %f secondi\n", milliseconds/1000.0f);
		    // Usa il conteggio nz della matrice CSR per i confronti
		    double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
		    printf("Tempo esecuzione parallelo HLL: %f secondi\n", milliseconds/1000.0f);
		    printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
		    print_flops(flops_parallel);
			printf("Errore relativo: %f\n", diffMetrics.mean_rel_err);
            printf("Errore assoluto: %f\n", diffMetrics.mean_abs_err);
			}
		}
		PerformanceMetrics perf_metrics_hll_naive;
		// Calcola e stampa il tempo medio in secondi
		perf_metrics_hll_naive.time = get_metric_value(ROW_HLL_TIME);
		printf("Tempo medio impiegato per l'esecuzione del Kernel-HLL-Naive: %f secondi\n", perf_metrics_hll_naive.time);
		perf_metrics_hll_naive.flops = calculate_flops(csr_mat.nz, perf_metrics_hll_naive.time);
        printf("Flops medi: ");
        print_flops(perf_metrics_hll_naive.flops);
		DiffMetrics mediumHllNaive = computeAverageErrors(ROW_HLL_TIME);
        printf("Errore relativo medio parallelo HLL row: %f\n", mediumHllNaive.mean_rel_err);
        printf("Errore assoluto medio parallelo HLL row: %f\n", mediumHllNaive.mean_abs_err);

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                  spmv_hll_warp_kernel, 0, 0);

	// Assicurati che blockSize sia multiplo di 32 (dimensione di un warp)
	blockSize = (blockSize + 31 / 32) * 32;
	warpsPerBlock = blockSize / 32;
	int threadsPerBlock_hll_warp = blockSize;

	// Un warp per riga, quindi M warps totali
	int blocksPerGrid_hll_warp = (csr_mat.M + warpsPerBlock - 1) / warpsPerBlock;

    // =========================
    // TEST 2: VERSIONE HLL WARP
    // =========================
    printf("\n--- TEST WARP PER RIGA HLL ---\n");
    reset_medium_time_metrics();
    clear_gpu_cache(64);
    for(int i = 1; i < NUM_ITERATION; i++) {
        printf("Iterazione HLL-Warp numero: %d\n", i);
        checkCudaErrors(cudaMemset(d_y_hll, 0, csr_mat.M * sizeof(double)));
        checkCudaErrors(cudaEventRecord(start));

        spmv_hll_warp_kernel<<<blocksPerGrid_hll_warp, threadsPerBlock_hll_warp>>>(
            csr_mat.M,        // total_M
            d_blocks,         // Puntatore ai blocchi su GPU
            d_x_hll,          // Vettore x su GPU
            d_y_hll          // Vettore y su GPU
        );

       	checkCudaErrors(cudaGetLastError());

       	// Registra l'evento di fine
       	checkCudaErrors(cudaEventRecord(stop));

       	// Aspetta la fine dell'esecuzione
       	checkCudaErrors(cudaEventSynchronize(stop));

    	// Controlla la correttezza del risultato ad ogni iterazione
        checkCudaErrors(cudaMemcpy(y_hll, d_y_hll, csr_mat.M * sizeof(double), cudaMemcpyDeviceToHost));

    	printf("Verifica risultato iterazione %d:\n", i);
    	DiffMetrics diffMetrics = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M);
        accumulateErrors(&diffMetrics, WARP_HLL_TIME);
       	float milliseconds = 0;
       	checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
       	if(i > ITERATION_SKIP){
        	printf("Tempo Kernel-HLL-Warp: %f ms\n", milliseconds/1000.0f);
    		update_medium_metric(WARP_HLL_TIME, milliseconds / 1000.0f);
    		double flops_parallel = calculate_flops(csr_mat.nz, milliseconds/1000.0f);
    		printf("Tempo esecuzione parallelo HLL: %f secondi\n", milliseconds/1000.0f);
    		printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
    		print_flops(flops_parallel);
            printf("Errore relativo: %f\n", diffMetrics.mean_rel_err);
            printf("Errore assoluto: %f\n", diffMetrics.mean_abs_err);
        }
    }

	PerformanceMetrics perf_metrics_hll_warp;
    perf_metrics_hll_warp.time = get_metric_value(WARP_HLL_TIME);
    printf("Tempo medio Kernel-HLL-Warp: %f secondi\n", perf_metrics_hll_warp.time);
    perf_metrics_hll_warp.flops = calculate_flops(csr_mat.nz, perf_metrics_hll_warp.time);
    printf("Flops medi: ");
    print_flops(perf_metrics_hll_warp.flops);
    DiffMetrics mediumHllWarp = computeAverageErrors(WARP_HLL_TIME);
    printf("Errore relativo medio parallelo HLL Warp: %f\n", mediumHllWarp.mean_rel_err);
    printf("Errore assoluto medio parallelo HLL Warp: %f\n", mediumHllWarp.mean_abs_err);

    for (int b = 0; b < hll_mat.num_blocks; b++) {
    	checkCudaErrors(cudaFree(temp_blocks[b].JA));
    	checkCudaErrors(cudaFree(temp_blocks[b].AS));
    }

    // Output risultati medi e salvataggio CSV
    printf("\n========== Risultati medi per matrice %s ==========\n",
		dir->d_name);

    printf("--- Tempi di esecuzione ---\n");
    printf("Tempo medio seriale CSR: %f secondi\n", perf_metrics_serial_csr.time);
    printf("Tempo medio seriale HLL: %f secondi\n", perf_metrics_serial_hll.time);
    printf("Tempo medio CSR thread per riga: %f secondi\n", perf_metrics_naive_csr.time);
    printf("Tempo medio CSR parallelo warp per riga: %f secondi\n", perf_metrics_warp_csr.time);
    printf("Tempo medio CSR parallelo warp per riga con shared memory: %f secondi\n", perf_metrics_warp_shared_csr.time);
   	printf("Tempo medio HLL thread per riga: %f secondi\n", perf_metrics_hll_naive.time);
    printf("Tempo medio parallelo warp per riga HLL: %f secondi\n", perf_metrics_hll_warp.time);
    printf("Tempo medio parallelo warp per riga con Shared Memory HLL: %f secondi\n", perf_metrics_warp_shared_hll.time);

    printf("\n--- Prestazioni (FLOPS) ---\n");
    printf("Prestazioni seriali CSR: ");
    print_flops(perf_metrics_serial_csr.flops);
    printf("Prestazioni seriali HLL: ");
    print_flops(perf_metrics_serial_hll.flops);
    printf("Prestazioni parallele CSR THREAD PER RIGA: ");
    print_flops(perf_metrics_naive_csr.flops);
    printf("Prestazioni parallele CSR WARP PER RIGA: ");
    print_flops(perf_metrics_warp_csr.flops);
    printf("Prestazioni parallele CSR WARP PER RIGA CON SHARED MEMORY: ");
    print_flops(perf_metrics_warp_shared_csr.flops);
    printf("Prestazioni parallele HLL THREAD PER RIGA: ");
    print_flops(perf_metrics_hll_naive.flops);
    printf("Prestazioni parallele HLL WARP PER RIGA: ");
   	print_flops(perf_metrics_hll_warp.flops);
    printf("Prestazioni parallele HLL WARP PER RIGA CON SHARED MEM HLL: ");
    print_flops(perf_metrics_warp_shared_hll.flops);


    write_results_to_csv(dir->d_name, csr_mat.M, csr_mat.N, csr_mat.nz,
						perf_metrics_serial_csr.time, perf_metrics_serial_hll.time, perf_metrics_naive_csr.time,
						perf_metrics_warp_csr.time,perf_metrics_warp_shared_csr.time, perf_metrics_warp_shared_hll.time, perf_metrics_hll_naive.time, perf_metrics_hll_warp.time,
						perf_metrics_serial_csr.flops, perf_metrics_serial_hll.flops, perf_metrics_naive_csr.flops,
						perf_metrics_warp_csr.flops, perf_metrics_hll_naive.flops, perf_metrics_hll_warp.flops, perf_metrics_warp_shared_csr.flops, perf_metrics_warp_shared_hll.flops,
						mediumCsrParallel, mediumCsrWarp, mediumCsrWarpShared, mediumHllNaive, mediumHllWarp, mediumHllWarpShared,
                        output_csv);


    FREE_CHECK(temp_blocks);
    checkCudaErrors(cudaFree(d_blocks));
    checkCudaErrors(cudaFree(d_x_hll));
    checkCudaErrors(cudaFree(d_y_hll));
    FREE_CHECK(y_serial);
    FREE_CHECK(y_parallel_csr);
    FREE_CHECK(y_hll);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFree(d_row_ptr));
    checkCudaErrors(cudaFree(d_col_idx));
    checkCudaErrors(cudaFree(d_values));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

	}
}

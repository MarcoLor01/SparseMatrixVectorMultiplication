// src/main.c

#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <unistd.h>
#include "hll_matrix.h"
#include "matrix_parser.h"
#include "performance_calculate.h"
#include "utility.h"
#include <omp.h>
#include <csr_matrix.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

int num_threads[] = {2,4,8,16,32,40};
#define NUM_POSSIBLE_THREADS (sizeof(num_threads) / sizeof(num_threads[0]))
#define NUM_ITERATION 95 + ITERATION_SKIP

int main() {

    struct dirent *dir;
    const char *directory_path = "../matrix_for_test/";
    //const char *directory_path = "../matrix_generated/";
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
        if (convert_in_csr(&pre_mat, &csr_mat, dir->d_name) != 0) {
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

        // Allocazione memoria per vettori di input/output
        double *x = malloc(csr_mat.N * sizeof(double));
        double *y_serial = malloc(csr_mat.M * sizeof(double));
        double *y_parallel_csr = malloc(csr_mat.M * sizeof(double));
        double *y_hll = malloc(hll_mat.num_blocks * HACK_SIZE * sizeof(double));

        // Verifica corretta allocazione
        if (!x || !y_serial || !y_parallel_csr || !y_hll) {
            fprintf(stderr, "Errore nell'allocazione della memoria per i vettori\n");
            // Pulizia memoria
            FREE_CHECK(x);
            FREE_CHECK(y_serial);
            FREE_CHECK(y_parallel_csr);
            FREE_CHECK(y_hll);
            free_csr_matrix(&csr_mat);
            free_pre_matrix(&pre_mat);
            free_hll_matrix(&hll_mat);
            exit(EXIT_FAILURE);
        }

        // Inizializzazione vettore x
        init_vector_at_one(x, csr_mat.N);
        initialize_metrics();
        // ================================
        // TEST 1: VERSIONE SERIALE (CSR)
        // ================================
        printf("\n--- TEST SERIALE CSR ---\n");
        reset_medium_time_metrics();
        for (int j = 1; j <= NUM_ITERATION; j++) {
            printf("Iterazione numero: %d\n", j);
            memset(y_serial, 0, csr_mat.M * sizeof(double));

            double start = omp_get_wtime();
            csr_matrix_vector_mult(csr_mat.M, csr_mat.row_ptr, csr_mat.col_idx,
                                   csr_mat.values, x, y_serial);
            double end = omp_get_wtime();
            double time_serial = end - start;
            if(j > ITERATION_SKIP){
                update_medium_metric(SERIAL_TIME, time_serial);

                double flops_serial = calculate_flops(csr_mat.nz, time_serial);
                printf("Tempo esecuzione seriale, matrice in formato CSR: %f secondi\n", time_serial);
                printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_serial);
                print_flops(flops_serial);
            }
        }
        PerformanceMetrics perfMetricsCsr;
        perfMetricsCsr.time = get_metric_value(SERIAL_TIME);
        perfMetricsCsr.flops = calculate_flops(csr_mat.nz, perfMetricsCsr.time);
        printf("\nTempo medio seriale CSR: %f secondi\n", perfMetricsCsr.time);
        printf("FLOPS medi seriali CSR: ");
        print_flops(perfMetricsCsr.flops);


        // ================================
        // TEST 2: VERSIONE SERIALE HLL
        // ================================
        printf("\n--- TEST SERIALE HLL ---\n");
        clear_cache(64);
        reset_medium_time_metrics();
        for (int j = 1; j <= NUM_ITERATION; j++) {
            memset(y_hll, 0, hll_mat.num_blocks * HACK_SIZE * sizeof(double));

            double start = omp_get_wtime();
            spmv_hll_serial(hll_mat.num_blocks, hll_mat.blocks, x, y_hll);
            double end = omp_get_wtime();
            double time_serial_hll = end - start;

            DiffMetrics diffMetricsSerialHll = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M, 1e-5, 1e-4, true);
            accumulateErrors(&diffMetricsSerialHll, SERIAL_HLL_TIME);

            if(j > ITERATION_SKIP){
                update_medium_metric(SERIAL_HLL_TIME, time_serial_hll);
                double flops_hll_serial = calculate_flops(csr_mat.nz, time_serial_hll);
                printf("Tempo esecuzione seriale HLL: %f secondi\n", time_serial_hll);
                printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_hll_serial);
                print_flops(flops_hll_serial);
                printf("Errore relativo: %f\n", diffMetricsSerialHll.mean_rel_err);
                printf("Errore assoluto: %f\n", diffMetricsSerialHll.mean_abs_err);
            }
        }

        PerformanceMetrics perfMetricsHll;
        perfMetricsHll.time = get_metric_value(SERIAL_HLL_TIME);
        perfMetricsHll.flops = calculate_flops(csr_mat.nz, perfMetricsHll.time);
        printf("\nTempo medio seriale HLL: %f secondi\n", perfMetricsHll.time);
        printf("FLOPS medi seriali HLL: ");
        print_flops(perfMetricsHll.flops);
        DiffMetrics mediumHllError = computeAverageErrors(SERIAL_HLL_TIME);
        printf("\nErrore relativo medio seriale HLL: %f\n", mediumHllError.mean_rel_err);
        printf("\nErrore assoluto medio seriale HLL: %f\n", mediumHllError.mean_abs_err);



        // Ora eseguiamo tutti i test paralleli per ogni numero di thread
        for (int thread_idx = 0; thread_idx < NUM_POSSIBLE_THREADS; thread_idx++) {
            const char *output_csv = "../result/spmv_results_openmp.csv";
            int current_threads = num_threads[thread_idx];

            // Verifica che la matrice abbia dimensioni compatibili con il numero di thread
            if (pre_mat.M < current_threads || pre_mat.N < current_threads) {
                printf("Numero di righe inferiore al numero di thread (%d). Salto questa configurazione.\n", current_threads);
                continue;
            }

            printf("\n---- Test con %d thread ----\n", current_threads);

            // ================================
            // TEST 3: VERSIONE PARALLELA CSR
            // ================================
            printf("\n--- TEST PARALLELO CSR ---\n");
            clear_cache(64);
            reset_medium_time_metrics();
            int *thread_row_start = NULL;
            int *thread_row_end = NULL;

            int num_thread_used_csr = prepare_thread_distribution(
                csr_mat.M, csr_mat.row_ptr, current_threads, csr_mat.nz,
                &thread_row_start, &thread_row_end);

            for (int j = 1; j <= NUM_ITERATION; j++) {
                printf("Iterazione numero: %d\n", j);
                memset(y_parallel_csr, 0, csr_mat.M * sizeof(double));

                double start = omp_get_wtime();
                spvm_csr_parallel(csr_mat.row_ptr, csr_mat.col_idx, csr_mat.values,
                                 x, y_parallel_csr, num_thread_used_csr,
                                 thread_row_start, thread_row_end);
                double end = omp_get_wtime();
                double time_parallel_csr = end - start;
				DiffMetrics diffMetricsParallelCsr = computeDifferenceMetrics(y_serial, y_parallel_csr, csr_mat.M, 1e-5, 1e-4, true);
                accumulateErrors(&diffMetricsParallelCsr, PARALLEL_CSR_TIME);

                if(j > ITERATION_SKIP){
                    update_medium_metric(PARALLEL_CSR_TIME, time_parallel_csr);
                    double flops_parallel = calculate_flops(csr_mat.nz, time_parallel_csr);
                    double speedup_parallel = perfMetricsCsr.time / time_parallel_csr;
                    double efficiency_parallel = speedup_parallel / current_threads;

                    printf("Tempo esecuzione parallelo CSR: %f secondi\n", time_parallel_csr);
                    printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
                    print_flops(flops_parallel);
                    printf("Errore relativo: %f\n", diffMetricsParallelCsr.mean_rel_err);
                    printf("Errore assoluto: %f\n", diffMetricsParallelCsr.mean_abs_err);
                    printf("Valore dello speed-up: %f\n", speedup_parallel);
                    printf("Valore efficienza: %f\n", efficiency_parallel);
                }
            }
			PerformanceMetrics perfMetricsCsrParallel;
            perfMetricsCsrParallel.time = get_metric_value(PARALLEL_CSR_TIME);
            perfMetricsCsrParallel.speedup = perfMetricsCsr.time / perfMetricsCsrParallel.time;
            perfMetricsCsrParallel.efficiency = perfMetricsCsrParallel.speedup / current_threads;
            perfMetricsCsrParallel.flops = calculate_flops(csr_mat.nz, perfMetricsCsrParallel.time);

            printf("\nTempo medio parallelo CSR: %f secondi\n", perfMetricsCsrParallel.time);
            printf("Speedup medio: %f\n", perfMetricsCsrParallel.speedup);
            printf("Efficienza media: %f\n", perfMetricsCsrParallel.efficiency);
            printf("FLOPS medi: ");
            print_flops(perfMetricsCsrParallel.flops);
            DiffMetrics mediumCsrParallel = computeAverageErrors(PARALLEL_CSR_TIME);
            printf("Errore relativo medio parallelo CSR: %f\n", mediumCsrParallel.mean_rel_err);
            printf("Errore assoluto medio parallelo CSR: %f\n", mediumCsrParallel.mean_abs_err);


            // ================================
            // TEST 4: VERSIONE PARALLELA HLL
            // ================================

            printf("\n--- TEST PARALLELO HLL ---\n");
            clear_cache(64);
            reset_medium_time_metrics();
            int *thread_row_start_hll = NULL;
            int *thread_row_end_hll = NULL;
            int num_thread_used_hll = prepare_thread_distribution_hll(
                                    &hll_mat, current_threads, &thread_row_start_hll, &thread_row_end_hll);
            printf("Numero di thread usati: %d\n", num_thread_used_hll);
            for (int j = 1; j <= NUM_ITERATION; j++) {
                memset(y_hll, 0, hll_mat.num_blocks * HACK_SIZE * sizeof(double));

                double start = omp_get_wtime();
                spmv_hll(hll_mat.blocks, x, y_hll, num_thread_used_hll,
                         thread_row_start_hll, thread_row_end_hll);
                double end = omp_get_wtime();
                double time_parallel_hll = end - start;

				DiffMetrics diffMetricsParallelHll = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M, 1e-5, 1e-4, true);
                if(j > ITERATION_SKIP){
                    update_medium_metric(PARALLEL_HLL_TIME, time_parallel_hll);
                    double flops_hll = calculate_flops(csr_mat.nz, time_parallel_hll);
                    double speedup_hll = perfMetricsHll.time / time_parallel_hll;
                    double efficiency_hll = speedup_hll / current_threads;

                    printf("Tempo esecuzione parallelo HLL: %f secondi\n", time_parallel_hll);
                    printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_hll);
                    print_flops(flops_hll);
                    printf("Errore relativo: %f\n", diffMetricsParallelHll.mean_rel_err);
                    printf("Valore dello speed-up: %f\n", speedup_hll);
                    printf("Valore efficienza: %f\n", efficiency_hll);
                }
            }

			PerformanceMetrics perfMetricsHllParallel;
            perfMetricsHllParallel.time = get_metric_value(PARALLEL_HLL_TIME);
            perfMetricsHllParallel.speedup = perfMetricsHll.time / perfMetricsHllParallel.time;
            perfMetricsHllParallel.efficiency = perfMetricsHllParallel.speedup / current_threads;
            perfMetricsHllParallel.flops = calculate_flops(csr_mat.nz, perfMetricsHllParallel.time);

            printf("\nTempo medio parallelo HLL: %f secondi\n", perfMetricsHllParallel.time);
            printf("Speedup medio: %f\n", perfMetricsHllParallel.speedup);
            printf("Efficienza media: %f\n", perfMetricsHllParallel.efficiency);
            printf("FLOPS medi: ");
            print_flops(perfMetricsHllParallel.flops);
            DiffMetrics mediumHllParallel = computeAverageErrors(PARALLEL_HLL_TIME);
            printf("Errore relativo medio parallelo HLL: %f\n", mediumHllParallel.mean_rel_err);
            printf("Errore assoluto medio parallelo HLL: %f\n", mediumHllParallel.mean_abs_err);

            // ================================
            // TEST 5: VERSIONE PARALLELA HLL MIGLIORATA
            // ================================
            printf("\n--- TEST PARALLELO HLL MIGLIORATO---\n");
            clear_cache(64);
            reset_medium_time_metrics();

            for (int j = 1; j <= NUM_ITERATION; j++) {
                memset(y_hll, 0, hll_mat.num_blocks * HACK_SIZE * sizeof(double));

                double start = omp_get_wtime();
                spmv_hll_simd(hll_mat.blocks, x, y_hll, num_thread_used_hll,
                              thread_row_start_hll, thread_row_end_hll);
                double end = omp_get_wtime();
                double time_parallel_hll_simd = end - start;

                DiffMetrics diffMetricsHllSimd = computeDifferenceMetrics(y_serial, y_hll, csr_mat.M, 1e-5, 1e-4, true);
				accumulateErrors(&diffMetricsHllSimd, PARALLEL_HLL_SIMD_TIME);
                if(j > ITERATION_SKIP){

                    update_medium_metric(PARALLEL_HLL_SIMD_TIME, time_parallel_hll_simd);
                    double flops_hll_simd = calculate_flops(csr_mat.nz, time_parallel_hll_simd);
                    double speedup_hll_simd = perfMetricsHll.time / time_parallel_hll_simd;
                    double efficiency_hll_simd = speedup_hll_simd / current_threads;

                    printf("Tempo esecuzione parallelo HLL SIMD: %f secondi\n", time_parallel_hll_simd);
                    printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_hll_simd);
                    print_flops(flops_hll_simd);
                    printf("Errore relativo: %f\n", diffMetricsHllSimd.mean_rel_err);
                    printf("Valore dello speed-up: %f\n", speedup_hll_simd);
                    printf("Valore efficienza: %f\n", efficiency_hll_simd);

                }
            }

            PerformanceMetrics perfMetricsHllSimd;
            perfMetricsHllSimd.time = get_metric_value(PARALLEL_HLL_SIMD_TIME);
            perfMetricsHllSimd.speedup = perfMetricsHll.time / perfMetricsHllSimd.time;
            perfMetricsHllSimd.efficiency = perfMetricsHllSimd.speedup / current_threads;
            perfMetricsHllSimd.flops = calculate_flops(csr_mat.nz, perfMetricsHllSimd.time);

            printf("\nTempo medio parallelo HLL SIMD: %f secondi\n", perfMetricsHllSimd.time);
            printf("Speedup medio: %f\n", perfMetricsHllSimd.speedup);
            printf("Efficienza media: %f\n", perfMetricsHllSimd.efficiency);
            printf("FLOPS medi: ");
            print_flops(perfMetricsHllSimd.flops);
			DiffMetrics mediumHllSimd = computeAverageErrors(PARALLEL_HLL_SIMD_TIME);
            printf("Errore relativo medio parallelo HLL SIMD: %f\n", mediumHllSimd.mean_rel_err);
            printf("Errore assoluto medio parallelo HLL SIMD: %f\n", mediumHllSimd.mean_abs_err);

            // ================================
            // TEST 6: VERSIONE PARALLELA CSR MIGLIORATA
            // ================================
            printf("\n--- TEST PARALLELO CSR MIGLIORATA ---\n");
            clear_cache(64);
            reset_medium_time_metrics();

            for (int j = 1; j <= NUM_ITERATION; j++) {
                printf("Iterazione numero: %d\n", j);
                memset(y_parallel_csr, 0, csr_mat.M * sizeof(double));
                double start = omp_get_wtime();
                spvm_csr_parallel_simd(csr_mat.row_ptr, csr_mat.col_idx, csr_mat.values,
                                      x, y_parallel_csr, num_thread_used_csr,
                                      thread_row_start, thread_row_end);
                double end = omp_get_wtime();
                double time_parallel_simd = end - start;

                DiffMetrics diffMetricsParallelCsrSimd = computeDifferenceMetrics(y_serial, y_parallel_csr, csr_mat.M, 1e-5, 1e-4, true);
				accumulateErrors(&diffMetricsParallelCsrSimd, PARALLEL_SIMD_CSR_TIME);
                if(j > ITERATION_SKIP){
                    update_medium_metric(PARALLEL_SIMD_CSR_TIME, time_parallel_simd);
                    double flops_simd = calculate_flops(csr_mat.nz, time_parallel_simd);
                    double speedup_simd = perfMetricsCsr.time / time_parallel_simd;
                    double efficiency_simd = speedup_simd / current_threads;

                    printf("Tempo esecuzione parallelo CSR SIMD: %f secondi\n", time_parallel_simd);
                    printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_simd);
                    print_flops(flops_simd);
                    printf("Errore relativo: %f\n", diffMetricsParallelCsrSimd.mean_rel_err);
                    printf("Errore assoluto: %f\n", diffMetricsParallelCsrSimd.mean_abs_err);
                    printf("Valore dello speed-up: %f\n", speedup_simd);
                    printf("Valore efficienza: %f\n", efficiency_simd);
                }
            }

            PerformanceMetrics perfMetricsCsrSimd;
            perfMetricsCsrSimd.time = get_metric_value(PARALLEL_SIMD_CSR_TIME);
            perfMetricsCsrSimd.speedup = perfMetricsCsr.time / perfMetricsCsrSimd.time;
            perfMetricsCsrSimd.efficiency = perfMetricsCsrSimd.speedup / current_threads;
            perfMetricsCsrSimd.flops = calculate_flops(csr_mat.nz, perfMetricsCsrSimd.time);

            printf("\nTempo medio parallelo CSR SIMD: %f secondi\n", perfMetricsCsrSimd.time);
            printf("Speedup medio: %f\n", perfMetricsCsrSimd.speedup);
            printf("Efficienza media: %f\n", perfMetricsCsrSimd.efficiency);
            printf("FLOPS medi: ");
            print_flops(perfMetricsCsrSimd.flops);
			DiffMetrics mediumCsrSimdError = computeAverageErrors(PARALLEL_SIMD_CSR_TIME);
            printf("Errore relativo medio parallelo CSR SIMD: %f\n", mediumCsrSimdError.mean_rel_err);
            printf("Errore assoluto medio parallelo CSR SIMD: %f\n", mediumCsrSimdError.mean_abs_err);

            // Output risultati medi e salvataggio CSV
            printf("\n========== Risultati medi per matrice %s con %d thread ==========\n",
                   dir->d_name, current_threads);

            printf("--- Tempi di esecuzione ---\n");
            printf("Tempo medio seriale CSR: %f secondi\n", perfMetricsCsr.time);
            printf("Tempo medio seriale HLL: %f secondi\n", perfMetricsHll.time);
            printf("Tempo medio parallelo CSR: %f secondi\n", perfMetricsCsrParallel.time);
            printf("Tempo medio parallelo CSR SIMD: %f secondi\n", perfMetricsCsrSimd.time);
            printf("Tempo medio parallelo HLL: %f secondi\n", perfMetricsHllParallel.time);
            printf("Tempo medio parallelo HLL SIMD: %f secondi\n", perfMetricsHllSimd.time);

            printf("\n--- Speedup ---\n");
            printf("Speedup medio (parallelo CSR): %f\n", perfMetricsCsrParallel.speedup);
            printf("Speedup medio (parallelo CSR SIMD): %f\n", perfMetricsCsrSimd.speedup);
            printf("Speedup medio (parallelo HLL): %f\n", perfMetricsHllParallel.speedup);
            printf("Speedup medio (parallelo HLL SIMD): %f\n", perfMetricsHllSimd.speedup);

            printf("\n--- Efficienza ---\n");
            printf("Efficienza media (parallelo CSR): %f\n", perfMetricsCsrParallel.efficiency);
            printf("Efficienza media (parallelo CSR SIMD): %f\n", perfMetricsCsrSimd.efficiency);
            printf("Efficienza media (parallelo HLL): %f\n", perfMetricsHllParallel.efficiency);
            printf("Efficienza media (parallelo HLL SIMD): %f\n", perfMetricsHllSimd.efficiency);

            printf("\n--- Prestazioni (FLOPS) ---\n");
            printf("Prestazioni seriali CSR: ");
            print_flops(perfMetricsCsr.flops);
            printf("Prestazioni seriali HLL: ");
            print_flops(perfMetricsHll.flops);
            printf("Prestazioni parallele CSR: ");
            print_flops(perfMetricsCsrParallel.flops);
            printf("Prestazioni parallele CSR SIMD: ");
            print_flops(perfMetricsCsrSimd.flops);
            printf("Prestazioni parallele HLL: ");
            print_flops(perfMetricsHllParallel.flops);
            printf("Prestazioni parallele HLL SIMD: ");
            print_flops(perfMetricsHllSimd.flops);

            printf("\n--- Errore medio ---\n");
            printf("Errore medio relativo: %f e assoluto %f csr parallelo\n", mediumCsrParallel.mean_rel_err, mediumCsrParallel.mean_abs_err);
            printf("Errore medio relativo: %f e assoluto %f hll parallelo\n", mediumHllParallel.mean_rel_err, mediumHllParallel.mean_abs_err);
            printf("Errore medio relativo: %f e assoluto %f csr simd parallelo\n", mediumCsrSimdError.mean_rel_err, mediumCsrSimdError.mean_abs_err);
            printf("Errore medio relativo: %f e assoluto %f hll simd parallelo\n", mediumHllSimd.mean_rel_err, mediumHllSimd.mean_abs_err);


            // Salva risultati su CSV
            write_results_to_csv(
    		dir->d_name, csr_mat.M, csr_mat.N, csr_mat.nz,
    		current_threads,
    		perfMetricsCsr.time, perfMetricsHll.time, perfMetricsCsrParallel.time,
    		perfMetricsCsrSimd.time, perfMetricsHllParallel.time, perfMetricsHllSimd.time,
    		mediumCsrParallel, mediumHllParallel, mediumCsrSimdError, mediumHllSimd,
    		perfMetricsCsrParallel.speedup, perfMetricsCsrSimd.speedup, perfMetricsHllParallel.speedup, perfMetricsHllSimd.speedup,
    		perfMetricsCsrParallel.efficiency, perfMetricsCsrSimd.efficiency, perfMetricsHllParallel.efficiency, perfMetricsHllSimd.efficiency,
    		perfMetricsCsr.flops, perfMetricsHll.flops, perfMetricsCsrParallel.flops, perfMetricsCsrSimd.flops, perfMetricsHllParallel.flops, perfMetricsHllSimd.flops,
    		output_csv);


            // Liberiamo le risorse per questa configurazione di thread
            FREE_CHECK(thread_row_start);
            FREE_CHECK(thread_row_end);
            FREE_CHECK(thread_row_start_hll);
            FREE_CHECK(thread_row_end_hll);

        }

        printf("\n=========================================================\n");

        // Pulizia memoria
        cleanup_metrics();
        FREE_CHECK(x);
        FREE_CHECK(y_serial);
        FREE_CHECK(y_parallel_csr);
        FREE_CHECK(y_hll);
        free_csr_matrix(&csr_mat);
        free_pre_matrix(&pre_mat);
        free_hll_matrix(&hll_mat);
    }

    closedir(d);

}
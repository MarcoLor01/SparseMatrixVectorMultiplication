// src/main.c
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include "hll_matrix.h"
#include "matrix_parser.h"
#include "csr_matrix.h"
#include "performance_calculate.h"
#include "utility.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

int num_threads[] = {2, 4, 8, 16, 32, 40};
#define NUM_POSSIBLE_THREADS (sizeof(num_threads) / sizeof(num_threads[0]))
#define NUM_ITERATION 50

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
    int matrix_error = 0;
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
            printf("Errore nel processamento della matrice\n");
            continue;
        }

        print_pre_matrix(&pre_mat, false);

        // Conversione in formato CSR
        CSRMatrix csr_mat;
        if (convert_in_csr(&pre_mat, &csr_mat) != 0) {
            printf("Errore nella conversione in CSR\n");
            free_pre_matrix(&pre_mat);
            continue;
        }

        printf("Matrice: %s, Dimensione: %dx%d, Non-zero: %d\n",
               dir->d_name, csr_mat.M, csr_mat.N, csr_mat.nz);

        print_csr_matrix(&csr_mat);

        // Conversione in formato HLL/ELLPACK
        HLLMatrix hll_mat;
        if (convert_in_hll(&pre_mat, &hll_mat) != 0) {
            printf("Errore nella conversione in ELLPACK");
            free_csr_matrix(&csr_mat);
            free_pre_matrix(&pre_mat);
            continue;
        }

        // Allocazione memoria per vettori di input/output
        double *x = malloc(csr_mat.N * sizeof(double));
        double *y_serial = malloc(csr_mat.M * sizeof(double));
        double *y_parallel_csr = malloc(csr_mat.M * sizeof(double));
        double *y_parallel_better_csr = malloc(csr_mat.M * sizeof(double));
        double *y_parallel_hll = malloc(hll_mat.num_blocks * HACK_SIZE * sizeof(double));
        double *y_parallel_hll_simd = malloc(hll_mat.num_blocks * HACK_SIZE * sizeof(double));
        double *y_serial_hll = malloc(hll_mat.num_blocks * HACK_SIZE * sizeof(double));

        // Verifica corretta allocazione
        if (!x || !y_serial || !y_parallel_csr || !y_parallel_better_csr || !y_parallel_hll || !y_parallel_hll_simd || !y_serial_hll) {
            printf("Errore nell'allocazione della memoria per i vettori\n");
            // Pulizia memoria
            free(x);
            free(y_serial);
            free(y_parallel_csr);
            free(y_parallel_better_csr);
            free(y_parallel_hll);
            free(y_parallel_hll_simd);
            free(y_serial_hll);
            free_csr_matrix(&csr_mat);
            free_pre_matrix(&pre_mat);
            free_hll_matrix(&hll_mat);
            exit(EXIT_FAILURE);
        }

        // Inizializzazione vettore x
        for (int i = 0; i < csr_mat.N; i++) {
            x[i] = 1.0;
        }

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

            update_medium_metric(SERIAL_TIME, time_serial);

            double flops_serial = calculate_flops(csr_mat.nz, time_serial);
            printf("Tempo esecuzione seriale, matrice in formato CSR: %f secondi\n", time_serial);
            printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_serial);
            print_flops(flops_serial);
        }
        double avg_time_serial = get_metric_value(SERIAL_TIME) / NUM_ITERATION;
        double avg_flops_serial = calculate_flops(csr_mat.nz, avg_time_serial);
        printf("\nTempo medio seriale CSR: %f secondi\n", avg_time_serial);
        printf("FLOPS medi seriali CSR: ");
        print_flops(avg_flops_serial);

        // ================================
        // TEST 2: VERSIONE SERIALE HLL
        // ================================
        printf("\n--- TEST SERIALE HLL ---\n");
        reset_medium_time_metrics();
        for (int j = 1; j <= NUM_ITERATION; j++) {
            memset(y_serial_hll, 0, hll_mat.num_blocks * HACK_SIZE * sizeof(double));

            double start = omp_get_wtime();
            spmv_hll_serial(hll_mat.num_blocks, hll_mat.blocks, x, y_serial_hll);
            double end = omp_get_wtime();
            double time_serial_hll = end - start;
            update_medium_metric(SERIAL_HLL_TIME, time_serial_hll);

            double error_hll_serial = checkDifferences(y_serial_hll, y_serial, pre_mat.M, true);
            matrix_error += error_hll_serial != 0.0;

            double flops_hll_serial = calculate_flops(csr_mat.nz, time_serial_hll);
            printf("Tempo esecuzione seriale HLL: %f secondi\n", time_serial_hll);
            printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_hll_serial);
            print_flops(flops_hll_serial);
            printf("Errore relativo: %f\n", error_hll_serial);
        }

        double avg_time_serial_hll = get_metric_value(SERIAL_HLL_TIME) / NUM_ITERATION;
        double avg_flops_hll_serial = calculate_flops(csr_mat.nz, avg_time_serial_hll);
        printf("\nTempo medio seriale HLL: %f secondi\n", avg_time_serial_hll);
        printf("FLOPS medi seriali HLL: ");
        print_flops(avg_flops_hll_serial);

        // Ora eseguiamo tutti i test paralleli per ogni numero di thread
        for (int thread_idx = 0; thread_idx < NUM_POSSIBLE_THREADS; thread_idx++) {
            const char *output_csv = "../result/spmv_results.csv";
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
                update_medium_metric(PARALLEL_CSR_TIME, time_parallel_csr);

                double error_csr = checkDifferences(y_parallel_csr, y_serial, csr_mat.M, false);
                matrix_error += error_csr != 0.0;

                double flops_parallel = calculate_flops(csr_mat.nz, time_parallel_csr);
                double speedup_parallel = avg_time_serial / time_parallel_csr;
                double efficiency_parallel = speedup_parallel / current_threads;

                printf("Tempo esecuzione parallelo CSR: %f secondi\n", time_parallel_csr);
                printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_parallel);
                print_flops(flops_parallel);
                printf("Errore relativo: %f\n", error_csr);
                printf("Valore dello speed-up: %f\n", speedup_parallel);
                printf("Valore efficienza: %f\n", efficiency_parallel);
            }

            double avg_time_parallel_csr = get_metric_value(PARALLEL_CSR_TIME) / NUM_ITERATION;
            double avg_speedup_parallel_csr = avg_time_serial / avg_time_parallel_csr;
            double avg_efficiency_parallel_csr = avg_speedup_parallel_csr / current_threads;
            double avg_flops_parallel = calculate_flops(csr_mat.nz, avg_time_parallel_csr);
            printf("\nTempo medio parallelo CSR: %f secondi\n", avg_time_parallel_csr);
            printf("Speedup medio: %f\n", avg_speedup_parallel_csr);
            printf("Efficienza media: %f\n", avg_efficiency_parallel_csr);
            printf("FLOPS medi: ");
            print_flops(avg_flops_parallel);

            // ================================
            // TEST 4: VERSIONE PARALLELA HLL
            // ================================
            printf("\n--- TEST PARALLELO HLL ---\n");
            reset_medium_time_metrics();
            int *thread_row_start_hll = NULL;
            int *thread_row_end_hll = NULL;
            int num_thread_used_hll = prepare_thread_distribution_hll(
                                    &hll_mat, current_threads, &thread_row_start_hll, &thread_row_end_hll);
            for (int j = 1; j <= NUM_ITERATION; j++) {
                memset(y_parallel_hll, 0, hll_mat.num_blocks * HACK_SIZE * sizeof(double));

                double start = omp_get_wtime();
                spmv_hll(hll_mat.blocks, x, y_parallel_hll, num_thread_used_hll,
                         thread_row_start_hll, thread_row_end_hll);
                double end = omp_get_wtime();
                double time_parallel_hll = end - start;
                update_medium_metric(PARALLEL_HLL_TIME, time_parallel_hll);

                double error_hll = checkDifferences(y_parallel_hll, y_serial, pre_mat.M, true);
                matrix_error += error_hll != 0.0;

                double flops_hll = calculate_flops(csr_mat.nz, time_parallel_hll);
                double speedup_hll = avg_time_serial_hll / time_parallel_hll;
                double efficiency_hll = speedup_hll / current_threads;

                printf("Tempo esecuzione parallelo HLL: %f secondi\n", time_parallel_hll);
                printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_hll);
                print_flops(flops_hll);
                printf("Errore relativo: %f\n", error_hll);
                printf("Valore dello speed-up: %f\n", speedup_hll);
                printf("Valore efficienza: %f\n", efficiency_hll);
            }

            double avg_time_parallel_hll = get_metric_value(PARALLEL_HLL_TIME) / NUM_ITERATION;
            double avg_speedup_hll = avg_time_serial_hll / avg_time_parallel_hll;
            double avg_efficiency_hll = avg_speedup_hll / current_threads;
            double avg_flops_hll = calculate_flops(csr_mat.nz, avg_time_parallel_hll);
            printf("\nTempo medio parallelo HLL: %f secondi\n", avg_time_parallel_hll);
            printf("Speedup medio: %f\n", avg_speedup_hll);
            printf("Efficienza media: %f\n", avg_efficiency_hll);
            printf("FLOPS medi: ");
            print_flops(avg_flops_hll);

            // ================================
            // TEST 5: VERSIONE PARALLELA HLL MIGLIORATA
            // ================================
            printf("\n--- TEST PARALLELO HLL MIGLIORATO---\n");
            reset_medium_time_metrics();

            for (int j = 1; j <= NUM_ITERATION; j++) {
                memset(y_parallel_hll_simd, 0, hll_mat.num_blocks * HACK_SIZE * sizeof(double));

                double start = omp_get_wtime();
                spmv_hll_simd(hll_mat.blocks, x, y_parallel_hll_simd, num_thread_used_hll,
                              thread_row_start_hll, thread_row_end_hll);
                double end = omp_get_wtime();
                double time_parallel_hll_simd = end - start;
                update_medium_metric(PARALLEL_HLL_SIMD_TIME, time_parallel_hll_simd);

                double error_hll_simd = checkDifferences(y_parallel_hll_simd, y_serial, pre_mat.M, false);
                matrix_error += error_hll_simd != 0.0;

                double flops_hll_simd = calculate_flops(csr_mat.nz, time_parallel_hll_simd);
                double speedup_hll_simd = avg_time_serial_hll / time_parallel_hll_simd;
                double efficiency_hll_simd = speedup_hll_simd / current_threads;

                printf("Tempo esecuzione parallelo HLL SIMD: %f secondi\n", time_parallel_hll_simd);
                printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_hll_simd);
                print_flops(flops_hll_simd);
                printf("Errore relativo: %f\n", error_hll_simd);
                printf("Valore dello speed-up: %f\n", speedup_hll_simd);
                printf("Valore efficienza: %f\n", efficiency_hll_simd);
            }

            double avg_time_parallel_hll_simd = get_metric_value(PARALLEL_HLL_SIMD_TIME) / NUM_ITERATION;
            double avg_speedup_hll_simd = avg_time_serial_hll / avg_time_parallel_hll_simd;
            double avg_efficiency_hll_simd = avg_speedup_hll_simd / current_threads;
            double avg_flops_hll_simd = calculate_flops(csr_mat.nz, avg_time_parallel_hll_simd);
            printf("\nTempo medio parallelo HLL SIMD: %f secondi\n", avg_time_parallel_hll_simd);
            printf("Speedup medio: %f\n", avg_speedup_hll_simd);
            printf("Efficienza media: %f\n", avg_efficiency_hll_simd);
            printf("FLOPS medi: ");
            print_flops(avg_flops_hll_simd);

            // ================================
            // TEST 6: VERSIONE PARALLELA CSR MIGLIORATA
            // ================================
            printf("\n--- TEST PARALLELO CSR MIGLIORATA ---\n");
            reset_medium_time_metrics();

            for (int j = 1; j <= NUM_ITERATION; j++) {
                printf("Iterazione numero: %d\n", j);
                memset(y_parallel_better_csr, 0, csr_mat.M * sizeof(double));

                double start = omp_get_wtime();
                spvm_csr_parallel_simd(csr_mat.row_ptr, csr_mat.col_idx, csr_mat.values,
                                      x, y_parallel_better_csr, num_thread_used_csr,
                                      thread_row_start, thread_row_end);
                double end = omp_get_wtime();
                double time_parallel_simd = end - start;

                update_medium_metric(PARALLEL_SIMD_CSR_TIME, time_parallel_simd);

                double error_simd = checkDifferences(y_parallel_better_csr, y_serial, csr_mat.M, false);
                if (error_simd != 0) {
                    printf("ERRORE: Risultato CSR SIMD parallelo non corrisponde al risultato CSR seriale\n");
                    continue;
                }

                double flops_simd = calculate_flops(csr_mat.nz, time_parallel_simd);
                double speedup_simd = avg_time_serial / time_parallel_simd;
                double efficiency_simd = speedup_simd / current_threads;

                printf("Tempo esecuzione parallelo CSR SIMD: %f secondi\n", time_parallel_simd);
                printf("Valore dei FLOPS: %.2f\nValore formattato: ", flops_simd);
                print_flops(flops_simd);
                printf("Errore relativo: %f\n", error_simd);
                printf("Valore dello speed-up: %f\n", speedup_simd);
                printf("Valore efficienza: %f\n", efficiency_simd);
            }

            double avg_time_parallel_simd = get_metric_value(PARALLEL_SIMD_CSR_TIME) / NUM_ITERATION;
            double avg_speedup_simd = avg_time_serial / avg_time_parallel_simd;
            double avg_efficiency_simd = avg_speedup_simd / current_threads;
            double avg_flops_simd = calculate_flops(csr_mat.nz, avg_time_parallel_simd);
            printf("\nTempo medio parallelo CSR SIMD: %f secondi\n", avg_time_parallel_simd);
            printf("Speedup medio: %f\n", avg_speedup_simd);
            printf("Efficienza media: %f\n", avg_efficiency_simd);
            printf("FLOPS medi: ");
            print_flops(avg_flops_simd);

            // Output risultati medi e salvataggio CSV
            printf("\n========== Risultati medi per matrice %s con %d thread ==========\n",
                   dir->d_name, current_threads);

            printf("--- Tempi di esecuzione ---\n");
            printf("Tempo medio seriale CSR: %f secondi\n", avg_time_serial);
            printf("Tempo medio seriale HLL: %f secondi\n", avg_time_serial_hll);
            printf("Tempo medio parallelo CSR: %f secondi\n", avg_time_parallel_csr);
            printf("Tempo medio parallelo CSR SIMD: %f secondi\n", avg_time_parallel_simd);
            printf("Tempo medio parallelo HLL: %f secondi\n", avg_time_parallel_hll);
            printf("Tempo medio parallelo HLL SIMD: %f secondi\n", avg_time_parallel_hll_simd);

            printf("\n--- Speedup ---\n");
            printf("Speedup medio (parallelo CSR): %f\n", avg_speedup_parallel_csr);
            printf("Speedup medio (parallelo CSR SIMD): %f\n", avg_speedup_simd);
            printf("Speedup medio (parallelo HLL): %f\n", avg_speedup_hll);
            printf("Speedup medio (parallelo HLL SIMD): %f\n", avg_speedup_hll_simd);

            printf("\n--- Efficienza ---\n");
            printf("Efficienza media (parallelo CSR): %f\n", avg_efficiency_parallel_csr);
            printf("Efficienza media (parallelo CSR SIMD): %f\n", avg_efficiency_simd);
            printf("Efficienza media (parallelo HLL): %f\n", avg_efficiency_hll);
            printf("Efficienza media (parallelo HLL SIMD): %f\n", avg_efficiency_hll_simd);

            printf("\n--- Prestazioni (FLOPS) ---\n");
            printf("Prestazioni seriali CSR: ");
            print_flops(avg_flops_serial);
            printf("Prestazioni seriali HLL: ");
            print_flops(avg_flops_hll_serial);
            printf("Prestazioni parallele CSR: ");
            print_flops(avg_flops_parallel);
            printf("Prestazioni parallele CSR SIMD: ");
            print_flops(avg_flops_simd);
            printf("Prestazioni parallele HLL: ");
            print_flops(avg_flops_hll);
            printf("Prestazioni parallele HLL SIMD: ");
            print_flops(avg_flops_hll_simd);

            // Salva risultati su CSV
            write_results_to_csv(dir->d_name, csr_mat.M, csr_mat.N, csr_mat.nz,
                               current_threads, avg_time_serial, avg_time_serial_hll, avg_time_parallel_csr,
                               avg_time_parallel_simd, avg_time_parallel_hll, avg_time_parallel_hll_simd,
                               avg_speedup_parallel_csr, avg_speedup_simd, avg_speedup_hll, avg_speedup_hll_simd,
                               avg_efficiency_parallel_csr, avg_efficiency_simd, avg_efficiency_hll, avg_efficiency_hll_simd,
                               avg_flops_serial, avg_flops_hll_serial, avg_flops_parallel, avg_flops_simd, avg_flops_hll, avg_flops_hll_simd,
                               output_csv);

            // Liberiamo le risorse per questa configurazione di thread
            free(thread_row_start);
            free(thread_row_end);
            free(thread_row_start_hll);
            free(thread_row_end_hll);
        }

        printf("\n=========================================================\n");

        // Pulizia memoria
        free(x);
        free(y_serial);
        free(y_parallel_csr);
        free(y_parallel_better_csr);
        free(y_parallel_hll);
        free(y_parallel_hll_simd);
        free(y_serial_hll);
        free_csr_matrix(&csr_mat);
        free_pre_matrix(&pre_mat);
        free_hll_matrix(&hll_mat);
    }

    closedir(d);

    // Stampa delle matrici problematiche
    printf("\nMatrici che hanno prodotto errori: ");
}
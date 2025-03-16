//
// Created by HP on 09/03/2025.
//

#include <stdio.h>
#include <unistd.h>
#include "performance_calculate.h"

double metrics[NUM_METRICS] = {0.0};



// Funzione per ottenere il valore di una metrica
double get_metric_value(const MediumPerformanceMetric type) {
    return metrics[type];
}
// Funzione generica per aggiornare qualsiasi metrica
void update_medium_metric(const MediumPerformanceMetric type, const double value) {
    metrics[type] += value;
}

void reset_medium_time_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        metrics[i] = 0.0;
    }
}

// Funzione per calcolare i FLOPS
double calculate_flops(const int nz, const double time) {
    // 2 * NZ / tempo (in secondi) => risultato in FLOPS
    return 2.0 * nz / time;
}

// Funzione per convertire e stampare i FLOPS in formato leggibile
void print_flops(double flops) {
    const char *units[] = {"FLOPS", "KFLOPS", "MFLOPS", "GFLOPS", "TFLOPS", "PFLOPS", "EFLOPS"};
    int unit_index = 0;

    while (flops >= 1000.0 && unit_index < 6) {
        flops /= 1000.0;
        unit_index++;
    }

    printf("%.3f %s\n", flops, units[unit_index]);
}

// Funzione per calcolare e stampare le medie su tutte le matrici per un dato numero di thread
void calculate_and_print_thread_averages(int num_thread, const char *output_summary_file) {
    FILE *fp;
    int file_exists = access(output_summary_file, F_OK) == 0;

    fp = fopen(output_summary_file, "a");
    if (fp == NULL) {
        printf("Errore nell'apertura del file di riepilogo %s\n", output_summary_file);
        return;
    }

    // Scrivi l'intestazione se il file è nuovo
    if (!file_exists) {
        fprintf(fp, "num_threads,avg_speedup_parallel,avg_speedup_unrolling,avg_efficiency_parallel,avg_efficiency_unrolling,avg_flops_serial,avg_flops_parallel,avg_flops_unrolling\n");
    }

    // Apri il file dei risultati per leggere i dati
    FILE *fp_results = fopen("../result/spmv_results.csv", "r");
    if (fp_results == NULL) {
        printf("Errore nell'apertura del file dei risultati\n");
        fclose(fp);
        return;
    }

    char line[1024];
    int count = 0;
    double total_speedup_parallel = 0.0;
    double total_speedup_unrolling = 0.0;
    double total_efficiency_parallel = 0.0;
    double total_efficiency_unrolling = 0.0;
    double total_flops_serial = 0.0;
    double total_flops_parallel = 0.0;
    double total_flops_unrolling = 0.0;

    // Salta l'intestazione
    if (fgets(line, sizeof(line), fp_results) == NULL) {
        fclose(fp_results);
        fclose(fp);
        return;
    }

    // Leggi ogni riga
    while (fgets(line, sizeof(line), fp_results) != NULL) {
        char matrix_name[256];
        int rows, cols, nz, threads;
        double time_serial, time_parallel, time_parallel_unrolling;
        double flops_serial, flops_parallel, flops_parallel_unrolling;
        double speedup_parallel, speedup_unrolling, efficiency_parallel, efficiency_unrolling;

        // Estrai i dati dalla riga CSV
        sscanf(line, "%[^,],%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
               matrix_name, &rows, &cols, &nz, &threads,
               &time_serial, &time_parallel, &time_parallel_unrolling,
               &flops_serial, &flops_parallel, &flops_parallel_unrolling,
               &speedup_parallel, &speedup_unrolling,
               &efficiency_parallel, &efficiency_unrolling);

        // Controlla se è la riga per il numero di thread corrente
        if (threads == num_thread) {
            total_speedup_parallel += speedup_parallel;
            total_speedup_unrolling += speedup_unrolling;
            total_efficiency_parallel += efficiency_parallel;
            total_efficiency_unrolling += efficiency_unrolling;
            total_flops_serial += flops_serial;
            total_flops_parallel += flops_parallel;
            total_flops_unrolling += flops_parallel_unrolling;
            count++;
        }
    }

    fclose(fp_results);

    if (count > 0) {
        double avg_speedup_parallel = total_speedup_parallel / count;
        double avg_speedup_unrolling = total_speedup_unrolling / count;
        double avg_efficiency_parallel = total_efficiency_parallel / count;
        double avg_efficiency_unrolling = total_efficiency_unrolling / count;
        double avg_flops_serial = total_flops_serial / count;
        double avg_flops_parallel = total_flops_parallel / count;
        double avg_flops_unrolling = total_flops_unrolling / count;

        // Stampa a schermo
        printf("\n=== MEDIE PER %d THREAD SU TUTTE LE MATRICI (%d matrici) ===\n",
               num_thread, count);
        printf("Speedup medio (parallelo): %.4f\n", avg_speedup_parallel);
        printf("Speedup medio (unrolling): %.4f\n", avg_speedup_unrolling);
        printf("Efficienza media (parallelo): %.4f\n", avg_efficiency_parallel);
        printf("Efficienza media (unrolling): %.4f\n", avg_efficiency_unrolling);

        printf("FLOPS medi (seriale): %.4f\n", avg_flops_serial);
        printf("FLOPS medi (parallelo): %.4f\n", avg_flops_parallel);
        printf("FLOPS medi (unrolling): %.4f\n", avg_flops_unrolling);

        // Scrivi nel file di riepilogo
        fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
                num_thread, avg_speedup_parallel, avg_speedup_unrolling,
                avg_efficiency_parallel, avg_efficiency_unrolling,
                avg_flops_serial, avg_flops_parallel, avg_flops_unrolling);
    }

    fclose(fp);
}


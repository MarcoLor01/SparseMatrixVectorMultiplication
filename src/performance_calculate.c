//
// Created by HP on 09/03/2025.
//

#include <stdio.h>
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


//
// Created by HP on 09/03/2025.
//

#include <stdio.h>
#include "performance_calculate.cuh"

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


DifferenceMetrics computeDifferenceMetrics(
    const double* ref,
    const double* res,
    int n,
    double abs_tol,
    double rel_tol,
    bool print_summary
)
{
    DifferenceMetrics metrics = {0.0, 0.0};

    if (n <= 0) {
        if (print_summary) {
            printf("--- Comparison Summary ---\n");
            printf("Vector size        : 0\n");
            printf("Result             : PASS (empty vectors)\n");
            printf("------------------------\n");
        }
        return metrics;
    }

    double sum_abs_err = 0.0;
    double sum_rel_err = 0.0;

    for (int i = 0; i < n; ++i) {
        double abs_diff = std::fabs(ref[i] - res[i]);
        double ref_abs = std::fabs(ref[i]);

        sum_abs_err += abs_diff;

        if (ref_abs > abs_tol) {
            sum_rel_err += abs_diff / ref_abs;
        }
    }

    metrics.mean_abs_err = sum_abs_err / n;
    metrics.mean_rel_err = sum_rel_err / n;

    if (print_summary) {
        printf("--- Comparison Summary ---\n");
        printf("Vector size         : %d\n", n);
        printf("Mean Absolute Error : %.10e\n", metrics.mean_abs_err);
        printf("Mean Relative Error : %.10e\n", metrics.mean_rel_err);
        printf("------------------------\n");
    }

    return metrics;
}

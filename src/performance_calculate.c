//
// Created by HP on 09/03/2025.
//

#include <stdio.h>
#include "performance_calculate.h"
#include <stdlib.h>
#include <math.h>
#include <utility.h>

MetricStats metrics[NUM_METRICS];

void initialize_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        metrics[i].sum = 0.0;
        metrics[i].count = 0;
        metrics[i].capacity = INITIAL_CAPACITY;
        metrics[i].relative_error = 0.0;
        metrics[i].absolute_error = 0.0;
        metrics[i].values = (double*)malloc(INITIAL_CAPACITY * sizeof(double));
        if (!metrics[i].values) {
            fprintf(stderr, "Failed to allocate memory for metrics\n");
            exit(EXIT_FAILURE);
        }
    }
}

void cleanup_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        FREE_CHECK(metrics[i].values);
        metrics[i].values = NULL;
    }
}

// Funzione per ottenere il valore medio di una metrica
double get_metric_value(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].sum / metrics[type].count;
}

// Funzione per il calcolo dell'errore relativo
double get_relative_error(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].relative_error;
}

// Funzione per ottenere l'errore assoluto
double get_absolute_error(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].absolute_error;
}

void accumulateErrors(const DiffMetrics* iteration_metrics, const MediumPerformanceMetric type) {
    metrics[type].absolute_error += iteration_metrics->mean_abs_err;
    metrics[type].relative_error += iteration_metrics->mean_rel_err;
}

DiffMetrics computeAverageErrors(const MediumPerformanceMetric type) {
    DiffMetrics average = {0.0, 0.0};

    if (metrics[type].count > 0) {
        average.mean_abs_err = get_absolute_error(type) / (metrics[type].count + ITERATION_SKIP);
        average.mean_rel_err = get_relative_error(type) / (metrics[type].count + ITERATION_SKIP);
    }

    return average;
}

// Funzione per aggiornare una metrica
void update_medium_metric(const MediumPerformanceMetric type, const double value) {
    metrics[type].sum += value;

    if (metrics[type].count >= metrics[type].capacity) {
        metrics[type].capacity *= 2;
        metrics[type].values = (double*)realloc(metrics[type].values,
                                              metrics[type].capacity * sizeof(double));
        if (!metrics[type].values) {
            fprintf(stderr, "Failed to reallocate memory for metrics\n");
            exit(EXIT_FAILURE);
        }
    }

    // Aggiungi il nuovo valore all'array
    metrics[type].values[metrics[type].count] = value;
    metrics[type].count++;
}

void reset_medium_time_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        metrics[i].sum = 0.0;
        metrics[i].count = 0;
        metrics[i].relative_error = 0.0;
        metrics[i].absolute_error = 0.0;
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

struct DifferenceMetrics computeDifferenceMetrics(
    const double* ref,
    const double* res,
    int n,
    double abs_tol,
    double rel_tol,
    bool print_summary
)
{
    struct DifferenceMetrics metrics = {0.0, 0.0};

    if (n <= 0) {
        if (print_summary) {
            printf("--- Comparison Summary ---\n");
            printf("Vector size         : 0\n");
            printf("Result              : PASS (empty vectors)\n");
            printf("----------------------------\n");
        }
        return metrics;
    }

    double sum_abs_err = 0.0;
    double sum_rel_err = 0.0;

    for (int i = 0; i < n; ++i) {
        double abs_diff = fabs(ref[i] - res[i]);
        double ref_abs = fabs(ref[i]);

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
        printf("----------------------------\n");
    }

    return metrics;
}

